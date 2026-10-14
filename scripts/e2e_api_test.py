"""End-to-end client for a running PS-2 API: upload -> poll -> download -> verify.

    python scripts/e2e_api_test.py VIDEO [VIDEO ...] [--url http://127.0.0.1:8000]
                                   [--concurrent] [--out DIR] [--json report.json]

Only the video is sent (no backend or tracking parameters). For each job the script
checks that it reaches ``completed`` (or reports the failure message), downloads the
output video and decodes it with OpenCV to confirm it is readable and has the same
number of frames the job reports. Exit code 0 only if every job passed.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request


def _http(method: str, url: str, body: bytes = None, headers: dict = None, timeout=600):
    req = request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def upload(base: str, path: Path, extra_fields: dict = None):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = []
    for k, v in (extra_fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                     .encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
                 f"filename=\"{path.name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return _http("POST", f"{base}/track", b"".join(parts),
                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def run_one(base: str, video: Path, out_dir: Path, timeout_s: float) -> dict:
    t0 = time.perf_counter()
    status, body, _ = upload(base, video)
    rec = {"video": str(video), "upload_status": status}
    if status != 202:
        rec.update(passed=False, error=body.decode(errors="replace"))
        return rec
    job_id = json.loads(body)["job_id"]
    rec["job_id"] = job_id
    last = None
    while time.perf_counter() - t0 < timeout_s:
        _, b, _ = _http("GET", f"{base}/jobs/{job_id}")
        last = json.loads(b)
        if last["status"] in ("completed", "failed"):
            break
        time.sleep(1.0)
    rec["wall_time_s"] = round(time.perf_counter() - t0, 2)
    rec["job"] = last
    if not last or last["status"] != "completed":
        rec.update(passed=False, error=(last or {}).get("error") or "timed out")
        return rec
    _, sb, _ = _http("GET", f"{base}/jobs/{job_id}/summary")
    summary = json.loads(sb)
    rec["summary"] = summary
    vs, vb, vh = _http("GET", f"{base}/jobs/{job_id}/video")
    ext = Path(last.get("output_filename") or "tracked.mp4").suffix
    out = out_dir / f"{job_id}{ext}"
    out.write_bytes(vb)
    import cv2

    cap = cv2.VideoCapture(str(out))
    decoded = 0
    while cap.grab():
        decoded += 1
    cap.release()
    expected = summary["video"]["frame_count"]
    rec.update(output_file=str(out), output_http=vs, output_content_type=vh.get("content-type"),
               output_frames_decoded=decoded, frames_processed=expected,
               passed=(vs == 200 and decoded == expected and expected > 0))
    if not rec["passed"]:
        rec["error"] = f"output check failed: http {vs}, decoded {decoded} of {expected}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrent", action="store_true", help="submit all videos at once")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "e2e")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    st, hb, _ = _http("GET", f"{args.url}/health")
    health = json.loads(hb) if st == 200 else {"status": st}
    print("health:", json.dumps(health.get("reid", health), default=str)[:600])
    if args.concurrent:
        with ThreadPoolExecutor(len(args.videos)) as ex:
            results = list(ex.map(lambda v: run_one(args.url, v, args.out, args.timeout), args.videos))
    else:
        results = [run_one(args.url, v, args.out, args.timeout) for v in args.videos]
    for r in results:
        s = r.get("summary") or {}
        perf, trk = s.get("performance", {}), s.get("tracking", {})
        print(f"{'PASS' if r['passed'] else 'FAIL'} {Path(r['video']).name}: "
              f"status={(r.get('job') or {}).get('status')} frames={r.get('frames_processed')} "
              f"decoded={r.get('output_frames_decoded')} ids={trk.get('num_unique_track_ids')} "
              f"fps={perf.get('average_processing_fps')} backend={(perf.get('reid_backend') or {}).get('backend')} "
              f"wall={r.get('wall_time_s')}s err={r.get('error')}")
    if args.json:
        args.json.write_text(json.dumps({"health": health, "results": results}, indent=2, default=str))
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
