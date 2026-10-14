"""Regression tests for the backend layout: root worker, readiness route, path independence."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import worker
from source.config import settings
from tests.test_video_io import FakeRegistry, make_video

ROOT = Path(__file__).resolve().parents[1]


def test_paths_are_absolute_and_inside_project():
    assert settings.PROJECT_ROOT == ROOT
    for p in (settings.MODELS_DIR, settings.REID.onnx_path, settings.REID.tensorrt_dir,
              Path(settings.DETECTOR.weights), settings.APP.upload_dir, settings.APP.output_dir):
        assert p.is_absolute()
    assert settings.MODELS_DIR == ROOT / "source" / "models"


def test_dotenv_loader_never_overrides_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nPS2_TEST_A=from_file\nPS2_TEST_B='quoted'\nPS2_TEST_EMPTY=\n",
                   encoding="utf-8")
    monkeypatch.setenv("PS2_TEST_A", "from_env")
    monkeypatch.delenv("PS2_TEST_B", raising=False)
    monkeypatch.delenv("PS2_TEST_EMPTY", raising=False)
    settings.load_dotenv(env)
    import os
    assert os.environ["PS2_TEST_A"] == "from_env"
    assert os.environ["PS2_TEST_B"] == "quoted"
    assert "PS2_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("PS2_TEST_B")


def test_worker_rejects_missing_input(tmp_path, capsys):
    assert worker.main([str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")]) == 2
    assert "not found" in capsys.readouterr().err


def test_worker_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert worker.main([str(f), "--out", str(tmp_path / "o")]) == 2


def test_worker_job_on_invalid_video_fails_without_outputs(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    out = tmp_path / "out"
    state = worker.run_job(bad, out, registry=FakeRegistry())
    assert state["status"] == "failed" and "Invalid video" in state["error"]
    persisted = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert not list(out.glob("tracked.*")) and not (out / "tracks_mot.txt").exists()


def test_worker_job_success_uses_shared_pipeline(tmp_path):
    src = make_video(tmp_path / "in.mp4", n=12)
    out = tmp_path / "out"
    state = worker.run_job(src, out, registry=FakeRegistry())
    assert state["status"] == "completed", state["error"]
    persisted = json.loads((out / "job.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed" and persisted["frames_processed"] == 12
    assert (out / state["output_video"]).stat().st_size > 0
    assert (out / "tracks_mot.txt").is_file() and (out / "summary.json").is_file()


def test_ready_route_reflects_model_state(tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from source import main
    from source.logic.job_manager import JobManager

    cfg = replace(settings.APP, upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                  temp_dir=tmp_path / "u" / ".tmp")

    class NotLoaded(FakeRegistry):
        def status(self):
            return {**super().status(), "reid_loaded": False, "errors": {"reid": "x"}}

    for registry, code in ((FakeRegistry(), 200), (NotLoaded(), 503)):
        mgr = JobManager(cfg, registry=registry)
        main.app.state.jobs = mgr
        try:
            r = TestClient(main.app).get("/ready")
            assert r.status_code == code and r.json()["ready"] is (code == 200)
        finally:
            mgr.shutdown()


@pytest.mark.parametrize("cmd", [["source/main.py"], ["worker.py"]])
def test_entry_points_import_from_any_directory(tmp_path, cmd):
    """`--help`-style smoke: both entry points resolve the `source` package outside the root."""
    code = ("import runpy, sys; sys.argv=[sys.argv[0], '--help']; "
            f"runpy.run_path(r'{ROOT / cmd[0]}', run_name='not_main')")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_worker_without_video_starts_backend_server(monkeypatch):
    from source import main

    started = []
    monkeypatch.setattr(main, "run", lambda: started.append(True))
    assert worker.main([]) == 0 and started == [True]


def test_finished_job_survives_server_restart(tmp_path):
    """After a restart the old job ID still has status, video download and tracks."""
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from source import main
    from source.logic.job_manager import JobManager
    from tests.test_video_io import run_job

    cfg = replace(settings.APP, upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                  temp_dir=tmp_path / "u" / ".tmp")
    first = JobManager(cfg, registry=FakeRegistry())
    done = run_job(first, make_video(tmp_path / "in.mp4", n=10))
    interrupted = first.submit(first.new_job_id(), tmp_path / "never.mp4")
    first.shutdown()
    assert done.status.value == "completed"
    # simulate a job the previous server was still processing when it stopped
    import json as _json
    state = Path(interrupted.output_dir) / "job.json"
    data = _json.loads(state.read_text(encoding="utf-8"))
    data["status"] = "processing"
    state.write_text(_json.dumps(data), encoding="utf-8")

    second = JobManager(cfg, registry=FakeRegistry())  # "restarted server"
    assert second.restore_jobs() == 2
    main.app.state.jobs = second
    try:
        client = TestClient(main.app)
        assert client.get(f"/jobs/{done.job_id}").json()["status"] == "completed"
        r = client.get(f"/jobs/{done.job_id}/video")
        assert r.status_code == 200 and len(r.content) > 0
        assert "attachment" in r.headers["content-disposition"]  # Swagger shows "Download file"
        assert client.get(f"/jobs/{done.job_id}/tracks").status_code == 200
        body = client.get(f"/jobs/{interrupted.job_id}").json()
        assert body["status"] == "failed" and "restarted" in body["error"]
    finally:
        second.shutdown()


def test_video_endpoint_status_codes_and_openapi(tmp_path):
    """200 file download / 404 unknown or missing / 409 not ready or failed; binary in OpenAPI."""
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from source import main
    from source.logic.job_manager import Job, JobManager, JobStatus
    from tests.test_video_io import run_job

    cfg = replace(settings.APP, upload_dir=tmp_path / "u", output_dir=tmp_path / "o",
                  temp_dir=tmp_path / "u" / ".tmp")
    mgr = JobManager(cfg, registry=FakeRegistry())
    main.app.state.jobs = mgr
    try:
        client = TestClient(main.app)
        done = run_job(mgr, make_video(tmp_path / "in.mp4", n=10))
        r = client.get(f"/jobs/{done.job_id}/video")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert r.headers["content-disposition"].startswith("attachment")
        assert r.content == Path(done.output_path).read_bytes()

        assert client.get("/jobs/" + "0" * 32 + "/video").status_code == 404
        assert client.get("/jobs/not-a-job-id/video").status_code == 404

        for status, error in ((JobStatus.PROCESSING, None), (JobStatus.FAILED, "boom")):
            jid = mgr.new_job_id()
            mgr.jobs[jid] = Job(job_id=jid, input_path="x", output_dir=str(tmp_path / jid),
                                params={}, status=status, error=error)
            r = client.get(f"/jobs/{jid}/video")
            assert r.status_code == 409 and (error is None or error in r.json()["detail"])

        Path(done.output_path).unlink()  # completed, but the file is gone
        r = client.get(f"/jobs/{done.job_id}/video")
        assert r.status_code == 404 and "not found" in r.json()["detail"]

        spec = client.get("/openapi.json").json()["paths"]["/jobs/{job_id}/video"]["get"]
        content = spec["responses"]["200"]["content"]
        assert content["video/mp4"]["schema"] == {"type": "string", "format": "binary"}
        assert {"404", "409"} <= set(spec["responses"])
    finally:
        mgr.shutdown()


def test_docs_page_downloads_files_without_blob_buffering():
    """/docs is the custom Swagger page whose hooks avoid Chrome's Blob limit (Error: OK)."""
    from fastapi.testclient import TestClient

    from source import main

    client = TestClient(main.app)
    page = client.get("/docs")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for hook in ("attachmentAwareFetch", "DirectDownloadPlugin", "requestInterceptor",
                 '"/openapi.json"', "swagger-ui-bundle.js"):
        assert hook in page.text
    assert "/docs" not in client.get("/openapi.json").json()["paths"]
