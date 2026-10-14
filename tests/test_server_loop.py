"""Server event-loop robustness against clients that abort connections (Windows).

Background: on Windows uvicorn defaults to asyncio's Proactor loop, which logs
``ConnectionResetError [WinError 10054]`` from ``_call_connection_lost`` when a client
resets a connection and, when the reset hits during accept, closes the listening socket
(server refuses all further connections). ``python source/main.py`` uses the Selector loop.
"""
import asyncio
import logging
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from source import main

ROOT = Path(__file__).resolve().parents[1]
win_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows event-loop behaviour")


class _Handle:
    def __repr__(self):
        return "<Handle _ProactorBasePipeTransport._call_connection_lost()>"


def test_benign_disconnect_is_recognized():
    reset = ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
    assert main._is_benign_disconnect({"exception": reset, "handle": _Handle()})
    assert main._is_benign_disconnect({"exception": ConnectionAbortedError(), "handle": _Handle()})
    assert not main._is_benign_disconnect({"exception": reset, "handle": "<Handle other()>"})
    assert not main._is_benign_disconnect({"exception": ValueError("x"), "handle": _Handle()})


@win_only
def test_proactor_guard_drops_only_the_benign_traceback(caplog):
    loop = asyncio.ProactorEventLoop()
    try:
        caplog.set_level(logging.DEBUG)
        main._install_loop_guards(loop)
        assert any("Proactor event loop" in r.message for r in caplog.records)
        caplog.clear()
        loop.call_exception_handler({"message": "Exception in callback",
                                     "exception": ConnectionResetError(10054, "reset"),
                                     "handle": _Handle()})
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        loop.call_exception_handler({"message": "real problem", "exception": RuntimeError("boom")})
        assert [r for r in caplog.records if r.levelno >= logging.ERROR and "real problem" in r.message]
    finally:
        loop.close()


def test_guard_is_noop_on_selector_loop(caplog):
    loop = asyncio.SelectorEventLoop()
    try:
        main._install_loop_guards(loop)
        assert loop.get_exception_handler() is None
    finally:
        loop.close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path="/health", timeout=5):
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
        return s.recv(64).split(b" ")[1]


@win_only
@pytest.mark.slow
def test_server_keeps_accepting_after_many_aborted_connections(tmp_path):
    """Real server (python source/main.py, real models): 200 RST-aborted connections, then still OK."""
    port = _free_port()
    log = tmp_path / "server.log"
    env = dict(os.environ, PS2_PORT=str(port))
    with open(log, "wb") as fh:
        proc = subprocess.Popen([sys.executable, str(ROOT / "source" / "main.py")], cwd=ROOT / "tests", env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            try:
                if _get(port) == b"200":
                    break
            except OSError:
                pass
            assert proc.poll() is None and time.time() - t0 < 180, log.read_text(errors="replace")
            time.sleep(1)
        rnd = __import__("random").Random(0)
        for i in range(200):
            s = socket.create_connection(("127.0.0.1", port))
            if i % 3 == 1:
                s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            time.sleep(rnd.choice([0, 0.001, 0.01]))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            s.close()
        assert _get(port) == b"200"
        text = log.read_text(errors="replace")
        for bad in ("Accept failed", "ConnectionResetError", "Traceback"):
            assert bad not in text, text[-3000:]
    finally:
        proc.terminate()
        proc.wait(30)
