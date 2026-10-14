"""API + worker integration tests with the REAL detector and the auto-selected ReID
backend (TensorRT on a capable GPU machine, otherwise the best verified fallback)."""
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import source.config.settings as config
from source.config.settings import DETECTOR, REID

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "people-detection.mp4"

pytestmark = pytest.mark.skipif(
    not (REID.checkpoint.is_file() and Path(DETECTOR.weights).is_file()),
    reason="ReID checkpoint or detector weights missing")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from source import main

    tmp = tmp_path_factory.mktemp("api")
    app_cfg = replace(config.APP, upload_dir=tmp / "uploads", output_dir=tmp / "outputs",
                      temp_dir=tmp / "uploads" / ".tmp", max_upload_mb=50)
    main.app.state.app_config = app_cfg
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        del main.app.state.app_config


def wait_for(client, job_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/jobs/{job_id}").json()
        if r["status"] in ("completed", "failed"):
            return r
        time.sleep(0.5)
    raise TimeoutError(job_id)


def synthetic_video(path: Path, n=15):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(n):
        f = np.full((240, 320, 3), 40, np.uint8)
        cv2.rectangle(f, (10 + 5 * i, 50), (60 + 5 * i, 180), (0, 200, 0), -1)
        w.write(f)
    w.release()
    return path


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detector"]["available"] and body["reid"]["loaded"]
    sel = body["reid"]["selection"]
    assert sel["requested"] == "auto"
    assert sel["selected"]["backend"] == body["reid"]["backend"]["backend"]
    assert sel["attempts"][-1]["status"] == "selected"
    assert "pytorch-cpu" in body["installed_inference_backends"]
    assert ":" not in body["reid"]["checkpoint"]  # no absolute filesystem paths
    print("auto-selected ReID backend:", body["reid"]["backend"])


def test_rejects_unsupported_extension(client):
    r = client.post("/track", files={"video": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_video(client):
    r = client.post("/track", files={"video": ("broken.mp4", b"\x00" * 4096, "video/mp4")})
    assert r.status_code == 400 and "Invalid video" in r.json()["detail"]


def test_rejects_empty_file(client):
    r = client.post("/track", files={"video": ("empty.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_unknown_job_id(client):
    assert client.get("/jobs/" + "0" * 32).status_code == 404
    assert client.get("/jobs/../../etc").status_code == 404
    assert client.get("/jobs/" + "0" * 32 + "/video").status_code == 404


def test_synthetic_video_without_people(client, tmp_path):
    path = synthetic_video(tmp_path / "blocks.mp4")
    with open(path, "rb") as f:
        r = client.post("/track", files={"video": ("blocks.mp4", f, "video/mp4")})
    assert r.status_code == 202
    job = wait_for(client, r.json()["job_id"])
    assert job["status"] == "completed", job
    s = client.get(f"/jobs/{job['job_id']}/summary").json()
    assert s["video"]["frame_count"] == 15
    assert s["tracking"]["id_switches"] is None  # not fabricated without GT


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample video missing")
def test_full_flow_on_real_video(client):
    """Only the video is uploaded; the whole video is processed with the auto backend."""
    with open(SAMPLE, "rb") as f:
        r = client.post("/track", files={"video": (SAMPLE.name, f, "video/mp4")})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = wait_for(client, job_id)
    assert job["status"] == "completed", job
    assert job["progress_percent"] == 100.0 and job["output_available"]
    assert job["reid_backend"]["backend"] and job["processing_time_s"] > 0
    cap = cv2.VideoCapture(str(SAMPLE))
    n_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    s = client.get(f"/jobs/{job_id}/summary").json()
    assert s["video"]["frame_count"] == n_input == job["result"]["frames"]  # whole video
    assert s["video"]["stop_reason"] == "end_of_stream"
    assert s["tracking"]["num_unique_track_ids"] >= 1
    assert s["tracker_config"]["track_buffer"] == config.TRACKER.track_buffer
    assert s["performance"]["reid_backend"]["backend"] == job["reid_backend"]["backend"]
    v = client.get(f"/jobs/{job_id}/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    out = Path(job_id + Path(job["output_filename"]).suffix)
    try:
        out.write_bytes(v.content)
        cap = cv2.VideoCapture(str(out))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        assert n == n_input
    finally:
        out.unlink(missing_ok=True)
    rows = client.get(f"/jobs/{job_id}/tracks").text.splitlines()
    ids = {int(r.split(",")[1]) for r in rows}
    assert ids == set(s["tracking"]["unique_track_ids"])


def test_worker_failure_path_marks_job_failed(client, tmp_path):
    from source.main import app

    mgr = app.state.jobs
    job_id = mgr.new_job_id()
    missing = tmp_path / "gone.mp4"
    mgr.submit(job_id, missing)
    job = wait_for(client, job_id)
    assert job["status"] == "failed" and "not found" in job["error"]
    assert client.get(f"/jobs/{job_id}/video").status_code == 409
    assert client.get(f"/jobs/{job_id}/summary").status_code == 409
