"""Video writer selection, frame loop, output validation, cleanup and job-status tests.

These use a scripted fake detector / ReID backend, so they need no model weights.
"""
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import source.config.settings as config
from source.logic import video_processor
from source.utils import video_utils as video_io
from source.models.detector.inference import Detections, PersonDetector
from source.utils.video_utils import VideoWriterError, create_video_writer, validate_video_output
from source.logic.video_processor import VideoError, VideoProcessor, probe_video

W, H, FPS = 320, 240, 10.0
_RealWriter = cv2.VideoWriter
_RealCapture = cv2.VideoCapture
MP4V_ONLY = [("mp4v", ".mp4"), ("MJPG", ".avi")]
ON_WINDOWS = sys.platform == "win32"


# ----------------------------------------------------------------- fixtures
def make_video(path: Path, n=30, size=(W, H)) -> Path:
    w = _RealWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    assert w.isOpened()
    for i in range(n):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(f, (10 + 3 * (i % 80), 50), (60 + 3 * (i % 80), 200), (0, 200, 0), -1)
        w.write(f)
    w.release()
    return path


class FakeDetector(PersonDetector):
    """One person box moving right by 3 px/frame (or nothing when ``people=False``)."""

    name = "fake"

    def __init__(self, people=True):
        self.people = people

    def detect(self, frame, frame_index):
        if not self.people:
            return Detections.empty(frame_index)
        x = 10 + 3 * ((frame_index - 1) % 80)
        return Detections(np.array([[x, 50, x + 50, 200]], np.float32),
                          np.array([0.9], np.float32), frame_index)


class FakeBackend:
    def extract_embeddings(self, crops):
        v = np.ones((len(crops), config.REID.embedding_dim), np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def info(self):
        return {"backend": "fake", "device": "cpu", "precision": "fp32", "model": None}

    def close(self):
        pass


class FakeRegistry:
    def __init__(self, people=True):
        self.inference_lock = threading.Lock()
        self._det = FakeDetector(people)
        self.selection = {"requested": "auto", "selected": FakeBackend().info(), "attempts": []}

    def detector(self):
        return self._det

    def reid(self, backend=None):
        return FakeBackend()

    def status(self):
        return {"detector_loaded": True, "reid_loaded": True, "reid_backend": FakeBackend().info(),
                "reid_selection": self.selection, "errors": {}}

    def release(self):
        pass


def _fourcc_of(args):
    # cv2.VideoWriter(path, fourcc, fps, size) or (path, api, fourcc, fps, size)
    return args[1] if len(args) == 4 else args[2]


def refusing_writer(refused):
    """cv2.VideoWriter stand-in that fails to open for the given FOURCCs."""
    refused_codes = {cv2.VideoWriter_fourcc(*c) for c in refused}

    class _Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    def factory(*args):
        return _Closed() if _fourcc_of(args) in refused_codes else _RealWriter(*args)

    return factory


def garbage_writer(bad):
    """Writer that 'opens' fine for the given FOURCCs but produces an undecodable file."""
    bad_codes = {cv2.VideoWriter_fourcc(*c) for c in bad}

    class _Garbage:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return True

        def write(self, frame):
            with open(self.path, "ab") as f:
                f.write(b"\x00" * 64)

        def release(self):
            pass

        def getBackendName(self):
            return "fake"

    def factory(*args):
        return _Garbage(args[0]) if _fourcc_of(args) in bad_codes else _RealWriter(*args)

    return factory


def processor(people=True):
    return VideoProcessor(FakeDetector(people), FakeBackend(), config.TRACKER, config.REID)


# --------------------------------------------------------- writer selection
def test_mp4v_writer_initializes(tmp_path):
    writer, info = create_video_writer(tmp_path / "out.mp4", FPS, (W, H), MP4V_ONLY)
    try:
        assert info.codec == "mp4v" and info.path.suffix == ".mp4"
        for _ in range(5):
            writer.write(np.zeros((H, W, 3), np.uint8))
    finally:
        writer.release()
    assert validate_video_output(info.path, expected_frames=5, size=(W, H)) == 5
    assert not list(tmp_path.glob(".preflight*"))  # preflight scratch files removed


@pytest.mark.skipif(not ON_WINDOWS, reason="Media Foundation is Windows-only")
def test_default_is_browser_playable_h264_via_media_foundation(tmp_path):
    # Real encoder (no mocks): odd frame size on purpose.
    writer, info = create_video_writer(tmp_path / "out.mp4", 12.0, (321, 241))
    try:
        assert (info.codec, info.backend, info.path.suffix) == ("H264", "MSMF", ".mp4")
        for i in range(12):
            writer.write(np.full((241, 321, 3), i * 10, np.uint8))
    finally:
        writer.release()
    assert validate_video_output(info.path, expected_frames=12, size=(321, 241)) == 12
    cap = cv2.VideoCapture(str(info.path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    cap.release()
    assert fourcc in ("h264", "avc1")


def test_falls_back_to_mjpg_avi(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io.cv2, "VideoWriter", refusing_writer(["mp4v", "H264"]))
    writer, info = create_video_writer(tmp_path / "out.mp4", FPS, (W, H))
    try:
        assert info.codec == "MJPG" and info.path == tmp_path / "out.avi"
        writer.write(np.zeros((H, W, 3), np.uint8))
    finally:
        writer.release()
    assert validate_video_output(info.path) == 1
    assert not (tmp_path / "out.mp4").exists()


def test_preflight_skips_encoder_that_opens_but_writes_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io.cv2, "VideoWriter", garbage_writer(["H264"]))
    writer, info = create_video_writer(tmp_path / "out.mp4", FPS, (W, H))
    writer.release()
    assert info.codec == "mp4v"
    assert not list(tmp_path.glob(".preflight*"))


def test_no_supported_encoder_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io.cv2, "VideoWriter", refusing_writer(["mp4v", "MJPG", "H264"]))
    with pytest.raises(VideoWriterError, match="No supported video encoder"):
        create_video_writer(tmp_path / "out.mp4", FPS, (W, H))
    assert list(tmp_path.iterdir()) == []


def test_codec_spec_parsing():
    assert video_io.parse_codec_candidates("H264@msmf:.mp4, mp4v:.mp4, MJPG:.AVI") == (
        ("H264", ".mp4", "MSMF"), ("mp4v", ".mp4"), ("MJPG", ".avi"))
    for bad in ("h264", "avc1@NOPE:.mp4", "toolong:.mp4"):
        with pytest.raises(ValueError):
            video_io.parse_codec_candidates(bad)


def test_openh264_ffmpeg_path_not_in_defaults():
    # avc1/H264 through FFmpeg needs the unshipped openh264 DLL; only MSMF H.264 is allowed.
    for cand in video_io.parse_codec_candidates(config.APP.video_codecs):
        if cand[0].lower() in ("avc1", "h264"):
            assert len(cand) == 3 and cand[2] == "MSMF"


# -------------------------------------------------------- output validation
def test_validation_rejects_missing_empty_and_truncated(tmp_path):
    with pytest.raises(VideoWriterError, match="missing or empty"):
        validate_video_output(tmp_path / "nope.mp4")
    (tmp_path / "empty.mp4").write_bytes(b"")
    with pytest.raises(VideoWriterError, match="missing or empty"):
        validate_video_output(tmp_path / "empty.mp4")
    (tmp_path / "junk.mp4").write_bytes(b"\x00" * 2048)
    with pytest.raises(VideoWriterError):
        validate_video_output(tmp_path / "junk.mp4")
    good = make_video(tmp_path / "ten.mp4", n=10)
    with pytest.raises(VideoWriterError, match="incomplete"):
        validate_video_output(good, expected_frames=11)
    with pytest.raises(VideoWriterError, match="does not match"):
        validate_video_output(good, size=(W * 2, H))


# ------------------------------------------------------------- input probing
def test_probe_rejects_missing_empty_text_and_still_image(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "missing.mp4")
    (tmp_path / "empty.mp4").write_bytes(b"")
    with pytest.raises(VideoError, match="empty"):
        probe_video(tmp_path / "empty.mp4")
    (tmp_path / "text.mp4").write_text("not a video " * 200)
    with pytest.raises(VideoError, match="could not be opened"):
        probe_video(tmp_path / "text.mp4")
    ok, buf = cv2.imencode(".jpg", np.full((100, 100, 3), 128, np.uint8))
    (tmp_path / "photo.mp4").write_bytes(buf.tobytes())
    with pytest.raises(VideoError, match="single image"):
        probe_video(tmp_path / "photo.mp4")


# ------------------------------------------------------------- frame loop
def test_processes_every_frame(tmp_path):
    src = make_video(tmp_path / "in.mp4", n=30)
    s = processor().process(src, tmp_path / "tracked.mp4", tmp_path / "mot.txt")
    assert s["video"]["frame_count"] == 30 and s["video"]["stop_reason"] == "end_of_stream"
    assert s["output"]["frames_written"] == 30 == s["output"]["frames_decoded_on_validation"]
    assert s["output"]["filename"] == "tracked.mp4"
    assert s["detection"]["total_detections"] == 30
    assert s["tracking"]["num_unique_track_ids"] == 1 and s["tracking"]["no_ids_reason"] is None
    assert not s["video"]["ended_before_reported_length"]
    lines = (tmp_path / "mot.txt").read_text().splitlines()
    assert len(lines) == 30 - config.TRACKER.n_init + 1  # one row per confirmed frame
    assert src.stat().st_size > 0  # input untouched


def test_longer_video_is_processed_to_completion(tmp_path):
    src = make_video(tmp_path / "long.mp4", n=450)
    s = processor().process(src, tmp_path / "tracked.mp4", tmp_path / "mot.txt")
    assert s["video"]["frame_count"] == 450 == s["output"]["frames_decoded_on_validation"]
    assert s["video"]["stop_reason"] == "end_of_stream" and s["video"]["max_frames"] is None


def test_max_frames_is_reported(tmp_path):
    # Internal cap only (APP.max_frames); the API never sets it.
    src = make_video(tmp_path / "in.mp4", n=30)
    s = processor().process(src, tmp_path / "tracked.mp4", max_frames=1)
    assert s["video"]["frame_count"] == 1 and s["video"]["stop_reason"] == "max_frames"
    assert s["video"]["reported_frame_count"] == 30
    assert s["tracking"]["num_unique_track_ids"] == 0
    assert "n_init" in s["tracking"]["no_ids_reason"]


class _FlakyCapture:
    fail_at = 6
    raise_error = False

    def __init__(self, path, *a):
        self._cap, self._reads = _RealCapture(path, *a), 0
        self._flaky = Path(path).name == "in.mp4"

    def read(self):
        self._reads += 1
        if self._flaky and self._reads == self.fail_at:
            if self.raise_error:
                raise cv2.error("simulated decoder error")
            return False, None
        return self._cap.read()

    def __getattr__(self, name):
        return getattr(self._cap, name)


@pytest.mark.parametrize("raise_error", [False, True])
def test_unreadable_frame_is_skipped_not_fatal(tmp_path, monkeypatch, raise_error):
    src = make_video(tmp_path / "in.mp4", n=20)
    flaky = type("Flaky", (_FlakyCapture,), {"raise_error": raise_error, "fail_at": 6})
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", flaky)
    s = processor().process(src, tmp_path / "tracked.mp4")
    assert s["video"]["frame_count"] == 20
    assert s["video"]["skipped_unreadable_frames"] == 1


def test_truncated_input_is_processed_and_flagged(tmp_path):
    # An interrupted copy/upload: the AVI header still claims 60 frames, the data ends early.
    src = tmp_path / "full.avi"
    w = _RealWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(60):
        w.write(np.full((H, W, 3), (i * 4) % 255, np.uint8))
    w.release()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(src.read_bytes()[: src.stat().st_size // 2])
    s = processor().process(cut, tmp_path / "tracked.mp4")
    assert 0 < s["video"]["frame_count"] < 60
    assert s["video"]["reported_frame_count"] == 60
    assert s["video"]["ended_before_reported_length"] is True
    assert s["output"]["frames_decoded_on_validation"] == s["video"]["frame_count"]


def test_no_people_is_not_a_tracking_failure(tmp_path):
    src = make_video(tmp_path / "in.mp4", n=10)
    s = processor(people=False).process(src, tmp_path / "tracked.mp4")
    assert s["video"]["frame_count"] == 10
    assert s["detection"]["total_detections"] == 0
    assert s["tracking"]["num_unique_track_ids"] == 0
    assert "no detectable people" in s["tracking"]["no_ids_reason"]


@pytest.mark.parametrize("payload", [b"", b"\x00" * 4096])
def test_empty_or_corrupt_input(tmp_path, payload):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(payload)
    with pytest.raises(VideoError):
        processor().process(bad, tmp_path / "tracked.mp4")


def test_writer_failure_raises(tmp_path, monkeypatch):
    src = make_video(tmp_path / "in.mp4", n=5)
    monkeypatch.setattr(video_io.cv2, "VideoWriter", refusing_writer(["mp4v", "MJPG", "H264"]))
    with pytest.raises(VideoWriterError):
        processor().process(src, tmp_path / "tracked.mp4")


# ------------------------------------------------------- job status + API
@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient

    from source import main
    from source.logic.job_manager import JobManager

    cfg = replace(config.APP, upload_dir=tmp_path / "uploads", output_dir=tmp_path / "outputs",
                  temp_dir=tmp_path / "uploads" / ".tmp")
    mgr = JobManager(cfg, registry=FakeRegistry())
    client = TestClient(main.app)  # no lifespan: real models are never loaded
    main.app.state.jobs = mgr  # routes read the config from the job manager
    yield client, mgr, tmp_path
    mgr.shutdown()


def run_job(mgr, src, params=None, timeout=120):
    job = mgr.submit(mgr.new_job_id(), src, params or {})
    return wait_job(job, timeout)


def wait_job(job, timeout=120):
    t0 = time.time()
    while job.status.value not in ("completed", "failed"):
        assert time.time() - t0 < timeout, f"job stuck in {job.status.value}"
        time.sleep(0.05)
    return job


def test_job_completed_with_validated_output(api):
    client, mgr, tmp = api
    src = make_video(tmp / "in.mp4", n=12)
    job = run_job(mgr, src)
    assert job.status.value == "completed" and job.error is None
    assert job.output_video.name == "tracked.mp4"
    r = client.get(f"/jobs/{job.job_id}").json()
    assert r["status"] == "completed" and r["output_available"] and r["frames_processed"] == 12
    assert r["output_filename"] == "tracked.mp4" and r["stop_reason"] == "end_of_stream"
    assert r["reid_backend"]["backend"] == "fake" and r["processing_time_s"] > 0
    assert r["result"]["frames"] == 12 and r["result"]["num_unique_track_ids"] == 1
    v = client.get(f"/jobs/{job.job_id}/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    assert len(v.content) == job.output_video.stat().st_size
    t = client.get(f"/jobs/{job.job_id}/tracks")
    assert t.status_code == 200 and t.text.startswith(f"{config.TRACKER.n_init},1,")
    assert client.get(f"/jobs/{job.job_id}/summary").json()["video"]["frame_count"] == 12
    assert not src.exists()  # input removed once the job finished


def test_job_avi_fallback_updates_output_path(api, monkeypatch):
    client, mgr, tmp = api
    src = make_video(tmp / "in.mp4", n=8)
    monkeypatch.setattr(video_io.cv2, "VideoWriter", refusing_writer(["mp4v", "H264"]))
    job = run_job(mgr, src)
    assert job.status.value == "completed", job.error
    assert job.output_codec == "MJPG" and job.output_video.suffix == ".avi"
    assert not job.requested_output.exists()
    r = client.get(f"/jobs/{job.job_id}").json()
    assert r["output_filename"] == "tracked.avi" and r["output_codec"] == "MJPG"
    v = client.get(f"/jobs/{job.job_id}/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/x-msvideo"
    assert "tracked_" in v.headers["content-disposition"] and ".avi" in v.headers["content-disposition"]


def test_job_failed_when_no_encoder(api, monkeypatch):
    client, mgr, tmp = api
    src = make_video(tmp / "in.mp4", n=5)
    monkeypatch.setattr(video_io.cv2, "VideoWriter", refusing_writer(["mp4v", "MJPG", "H264"]))
    job = run_job(mgr, src)
    assert job.status.value == "failed" and "No supported video encoder" in job.error
    assert job.output_path is None
    r = client.get(f"/jobs/{job.job_id}").json()
    assert r["status"] == "failed" and not r["output_available"] and r["error"]
    assert client.get(f"/jobs/{job.job_id}/video").status_code == 409
    assert client.get(f"/jobs/{job.job_id}/summary").status_code == 409
    assert client.get(f"/jobs/{job.job_id}/tracks").status_code == 409


def test_job_failed_when_output_does_not_validate(api, monkeypatch):
    client, mgr, tmp = api
    src = make_video(tmp / "in.mp4", n=5)

    def broken(path, expected_frames=None, size=None):
        raise VideoWriterError("Output video contains no decodable frames.")

    monkeypatch.setattr(video_processor, "validate_video_output", broken)
    job = run_job(mgr, src)
    assert job.status.value == "failed" and "no decodable frames" in job.error
    assert client.get(f"/jobs/{job.job_id}/video").status_code == 409
    # partial artifacts are removed; only job.json remains
    assert [p.name for p in Path(job.output_dir).iterdir()] == ["job.json"]
    assert not src.exists()


def test_unexpected_worker_exception_fails_job_without_leaking(api, monkeypatch):
    client, mgr, tmp = api
    src = make_video(tmp / "in.mp4", n=5)

    def boom(self, frame, frame_index):
        raise ValueError(r"secret C:\internal\path")

    monkeypatch.setattr(FakeDetector, "detect", boom)
    job = run_job(mgr, src)
    assert job.status.value == "failed"
    assert "ValueError" in job.error and "internal" not in job.error


def test_model_init_failure_gives_actionable_error(api, monkeypatch):
    from source.models.reid.inference import ReIDModelError

    client, mgr, tmp = api

    def no_backend(backend=None):
        raise ReIDModelError("No compatible ReID backend could be loaded: onnx-cpu: missing")

    monkeypatch.setattr(mgr.registry, "reid", no_backend)
    job = run_job(mgr, make_video(tmp / "in.mp4", n=5))
    assert job.status.value == "failed"
    assert job.error.startswith("Model initialization failed: No compatible ReID backend")


def test_missing_input_file_fails_cleanly(api):
    client, mgr, tmp = api
    job = run_job(mgr, tmp / "gone.mp4")
    assert job.status.value == "failed" and "not found" in job.error


def test_job_on_video_without_people(api, monkeypatch):
    client, mgr, tmp = api
    monkeypatch.setattr(mgr.registry._det, "people", False)
    job = run_job(mgr, make_video(tmp / "in.mp4", n=6))
    assert job.status.value == "completed"
    s = client.get(f"/jobs/{job.job_id}/summary").json()
    assert s["tracking"]["num_unique_track_ids"] == 0 and s["detection"]["total_detections"] == 0


def test_repeated_jobs_are_independent(api):
    client, mgr, tmp = api
    results = []
    for i in range(3):
        job = run_job(mgr, make_video(tmp / f"in{i}.mp4", n=15))
        assert job.status.value == "completed", job.error
        results.append(client.get(f"/jobs/{job.job_id}/summary").json())
    assert {r["video"]["frame_count"] for r in results} == {15}
    assert {r["tracking"]["num_unique_track_ids"] for r in results} == {1}  # no state leak


def test_upload_only_video_through_api_and_poll(api):
    client, mgr, tmp = api
    src = make_video(tmp / "clip.mp4", n=20)
    with open(src, "rb") as f:
        r = client.post("/track", files={"video": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["reid_backend"]["backend"] == "fake"
    assert body["video"]["frame_count"] == 20 and body["tracks_url"].endswith("/tracks")
    job = wait_job(mgr.get(body["job_id"]))
    st = client.get(f"/jobs/{job.job_id}").json()
    assert st["status"] == "completed" and st["frames_processed"] == 20
    assert st["input_filename"] == "clip.mp4"
    assert list((tmp / "uploads").glob("*.mp4")) == []  # upload cleaned up


def test_legacy_tuning_fields_are_ignored(api):
    """Old clients (and Swagger pre-fill) sent max_frames=1 etc.; the whole video is processed."""
    client, mgr, tmp = api
    src = make_video(tmp / "clip.mp4", n=20)
    with open(src, "rb") as f:
        r = client.post("/track", files={"video": ("clip.mp4", f, "video/mp4")},
                        data={"max_frames": "1", "reid_backend": "tensorrt",
                              "detection_confidence": "0.05", "track_buffer": "3000",
                              "reassociation_threshold": "0.05", "tracker_mode": "baseline"})
    assert r.status_code == 202
    job = wait_job(mgr.get(r.json()["job_id"]))
    s = client.get(f"/jobs/{job.job_id}/summary").json()
    assert s["video"]["frame_count"] == 20 and s["video"]["stop_reason"] == "end_of_stream"
    assert s["tracker_config"] == {k: v for k, v in
                                   __import__("dataclasses").asdict(config.TRACKER).items()}


def test_openapi_exposes_only_the_video_field(api):
    client, _, _ = api
    spec = client.get("/openapi.json").json()
    ref = spec["paths"]["/track"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    name = ref["$ref"].split("/")[-1]
    assert set(spec["components"]["schemas"][name]["properties"]) == {"video"}


@pytest.mark.parametrize("name,payload,code,detail", [
    ("notes.txt", b"hello", 415, "Unsupported file type"),
    ("noext", b"hello", 415, "Unsupported file type"),
    ("empty.mp4", b"", 400, "empty"),
    ("broken.mp4", b"\x00" * 4096, 400, "Invalid video"),
])
def test_invalid_uploads_rejected_with_clear_errors(api, name, payload, code, detail):
    client, mgr, tmp = api
    r = client.post("/track", files={"video": (name, payload, "application/octet-stream")})
    assert r.status_code == code and detail in r.json()["detail"]
    assert list((tmp / "uploads").glob("*.*")) == []  # nothing left behind
    assert mgr.pending_count() == 0


def test_concurrent_submissions_all_complete(api):
    client, mgr, tmp = api
    srcs = [make_video(tmp / f"c{i}.mp4", n=25) for i in range(4)]
    codes = []

    def submit(p):
        with open(p, "rb") as f:
            codes.append(client.post("/track", files={"video": (p.name, f, "video/mp4")}))

    threads = [threading.Thread(target=submit, args=(p,)) for p in srcs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [c.status_code for c in codes] == [202] * 4
    jobs = [wait_job(mgr.get(c.json()["job_id"])) for c in codes]
    assert all(j.status.value == "completed" for j in jobs), [j.error for j in jobs]
    assert all(j.frames_processed == 25 for j in jobs)
    assert len({j.output_path for j in jobs}) == 4


def test_stale_files_are_swept(api):
    client, mgr, tmp = api
    (tmp / "uploads").mkdir(exist_ok=True)
    (tmp / "uploads" / ".tmp").mkdir(exist_ok=True)
    import os

    week_ago = time.time() - 7 * 86400
    orphan = tmp / "uploads" / ("a" * 32 + ".mp4")
    orphan.write_bytes(b"x")
    os.utime(orphan, (week_ago, week_ago))
    recent = tmp / "uploads" / ("c" * 32 + ".mp4")  # may belong to another live server
    recent.write_bytes(b"x")
    (tmp / "uploads" / ".tmp" / "spool123").write_bytes(b"x")
    old = tmp / "outputs" / ("b" * 32)
    old.mkdir(parents=True)
    os.utime(old, (week_ago, week_ago))
    counts = mgr.sweep_stale_files()
    assert counts == {"uploads": 1, "temp": 1, "outputs": 1}
    assert not old.exists() and not orphan.exists() and recent.exists()


def test_truncated_input_completes_with_warning_in_job_status(api):
    client, mgr, tmp = api
    src = tmp / "full.avi"
    w = _RealWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(60):
        w.write(np.full((H, W, 3), (i * 4) % 255, np.uint8))
    w.release()
    cut = tmp / "cut.avi"
    cut.write_bytes(src.read_bytes()[: src.stat().st_size // 2])
    job = run_job(mgr, cut)
    assert job.status.value == "completed", job.error
    warnings = client.get(f"/jobs/{job.job_id}").json()["result"]["warnings"]
    assert len(warnings) == 1 and "truncated" in warnings[0]
