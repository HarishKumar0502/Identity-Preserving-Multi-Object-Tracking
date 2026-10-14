"""FastAPI application entry point for PS-2 occlusion-resistant multi-object tracking.

Run (from any directory):
       python source/main.py                 (host/port: PS2_HOST / PS2_PORT, default 127.0.0.1:8000)
  or:  python -m uvicorn source.main:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
       (uvicorn form: run from the project root)
Open:  http://127.0.0.1:8000/docs

On Windows the server must run on the asyncio *Selector* event loop. uvicorn's default
there is the Proactor loop, which (CPython, verified with 3.14.7) logs
``ConnectionResetError: [WinError 10054]`` from ``_call_connection_lost`` whenever a
client aborts a connection, and - worse - closes the listening socket ("Accept failed on
a socket", WinError 64) when a client resets a connection during accept, after which
every request is refused. Browsers abort connections routinely (cancelled video
downloads, preconnect sockets, closed tabs).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: import the project as the `source` package
    _ROOT = str(Path(__file__).resolve().parents[1])
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import asyncio  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from source.api.docs import router as docs_router  # noqa: E402
from source.api.routes import router  # noqa: E402
from source.config.settings import APP, PROJECT_ROOT  # noqa: E402
from source.logic.job_manager import JobManager  # noqa: E402
from source.utils.logging_utils import configure_logging  # noqa: E402

configure_logging()
log = logging.getLogger("ps2.api")


def _is_benign_disconnect(context: dict) -> bool:
    """The Proactor-loop noise from a client that reset its connection (WinError 10054/10053)."""
    exc = context.get("exception")
    return (isinstance(exc, (ConnectionResetError, ConnectionAbortedError))
            and "_call_connection_lost" in repr(context.get("handle") or context.get("message", "")))


def _install_loop_guards(loop: asyncio.AbstractEventLoop):
    if sys.platform != "win32" or not isinstance(loop, getattr(asyncio, "ProactorEventLoop", ())):
        return
    log.warning("Running on the Windows Proactor event loop: a client that aborts a connection "
                "during accept can make it stop accepting connections. Start the server with "
                "'python source/main.py' or add '--loop asyncio:SelectorEventLoop' to uvicorn.")
    previous = loop.get_exception_handler()

    def handler(lp, context):
        if _is_benign_disconnect(context):
            log.debug("Client disconnected abruptly: %s", context.get("exception"))
            return
        if previous is not None:
            previous(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _install_loop_guards(asyncio.get_running_loop())
    cfg = getattr(app.state, "app_config", None) or APP
    # Spool multipart uploads next to the uploads folder rather than the system
    # %TEMP% (on this machine C: has < 3 GB free; a 500 MB upload is copied twice).
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cfg.temp_dir)
    manager = JobManager(cfg)
    app.state.jobs = manager
    manager.sweep_stale_files()
    manager.restore_jobs()  # finished jobs stay downloadable across restarts (until TTL)
    # Warm-load models (and auto-select + verify the ReID backend) so /health reflects
    # reality and the first job is not delayed.
    for loader in (manager.registry.detector, manager.registry.reid):
        try:
            loader()
        except Exception as exc:  # reported via /health and /ready; jobs will fail clearly
            log.error("Model preload failed: %s", exc)
    yield
    manager.shutdown()


app = FastAPI(
    title="PS-2: Identity-Preserving Multi-Object Tracking Through Dense Occlusion",
    description="Upload a video to `/track` (the only input), poll `/jobs/{job_id}`, then "
                "download `/jobs/{job_id}/video` and read `/jobs/{job_id}/summary`. The "
                "inference backend (TensorRT / ONNX Runtime / PyTorch, GPU or CPU) and all "
                "tracking parameters are chosen internally.",
    version="1.2.0",
    lifespan=lifespan,
    docs_url=None,  # served by source/api/docs.py (direct file downloads)
)
app.include_router(router)
app.include_router(docs_router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # never leak stack traces
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


def run():
    """Start the API on a loop that survives aborted client connections (see module doc)."""
    import uvicorn

    uvicorn.run("source.main:app", host=os.environ.get("PS2_HOST", "127.0.0.1"),
                port=int(os.environ.get("PS2_PORT", "8000")), app_dir=str(PROJECT_ROOT),
                loop="asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto")


if __name__ == "__main__":
    run()
