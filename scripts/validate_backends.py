"""Compare every available ReID backend against the PyTorch-CPU reference on real
person crops (from a video, if given) and report numerical agreement + speed.

    python scripts/validate_backends.py [--video samples/people-detection.mp4] [--n 64]
                                        [--fp16] [--json report.json]

Per backend: embedding shape, finiteness, min cosine / max |diff| vs the reference,
and whether ReID *matching* is preserved: the max change of any pairwise cosine
similarity and the nearest-neighbour agreement over all crops. ``--fp16`` also
builds (if needed) and checks the FP16 TensorRT engine. Exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from source.config.settings import REID  # noqa: E402
from source.logic.inference_manager import ORDER, default_candidates  # noqa: E402
from source.models.model_registry import create_backend  # noqa: E402

TOL = {"fp32": 0.99999, "fp16": 0.999}


def sample_crops(video: Path | None, n: int = 64):
    rng = np.random.default_rng(0)
    if video is None:
        return [rng.integers(0, 255, (int(rng.integers(80, 300)), int(rng.integers(30, 120)), 3),
                             dtype=np.uint8) for _ in range(n)]
    from source.models.detector.inference import create_detector

    det = create_detector()
    cap = cv2.VideoCapture(str(video))
    crops, idx = [], 0
    while len(crops) < n:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 5:
            continue
        d = det.detect(frame, idx)
        for (x1, y1, x2, y2), s in zip(d.boxes.astype(int), d.scores):
            if y2 - y1 > 40 and s >= 0.5:
                crops.append(frame[y1:y2, x1:x2].copy())
    cap.release()
    if not crops:
        raise SystemExit("No person crops found in the video.")
    return crops[:n]


def bench(backend, crops, reps=5):
    backend.extract_embeddings(crops)  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        out = backend.extract_embeddings(crops)
    return out, (time.perf_counter() - t) / reps


def compare(ref, out) -> dict:
    cos = (ref * out).sum(1)
    s_ref, s_out = ref @ ref.T, out @ out.T
    np.fill_diagonal(s_ref, -np.inf)
    np.fill_diagonal(s_out, -np.inf)
    nn_agree = float((s_ref.argmax(1) == s_out.argmax(1)).mean())
    np.fill_diagonal(s_ref, 0)
    np.fill_diagonal(s_out, 0)
    return {"shape": list(out.shape), "finite": bool(np.isfinite(out).all()),
            "min_cosine": float(cos.min()), "max_abs_diff": float(np.abs(ref - out).max()),
            "max_pairwise_similarity_change": float(np.abs(s_ref - s_out).max()),
            "nearest_neighbour_agreement": nn_agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="also validate the FP16 TensorRT engine")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    crops = sample_crops(args.video, args.n)
    print(f"{len(crops)} crops ({'video' if args.video else 'random'}), input {REID.input_size}")
    ref_backend = create_backend("pytorch-cpu")
    ref, t_ref = bench(ref_backend, crops)
    rows = [{"backend": "pytorch-cpu", "device": "cpu", "precision": "fp32",
             "ms_per_crop": 1000 * t_ref / len(crops), "reference": True}]
    print(f"{'pytorch-cpu':<13} [cpu] {rows[0]['ms_per_crop']:7.2f} ms/crop  (reference)")
    jobs = [(c.name, REID) for c in default_candidates(REID) if c.name != "pytorch-cpu"]
    if args.fp16:
        jobs.insert(1, ("tensorrt", replace(REID, tensorrt_precision="fp16")))
    failures = 0
    for name, cfg in jobs:
        label = name if cfg.tensorrt_precision == "fp32" or name != "tensorrt" else "tensorrt-fp16"
        cand = next(c for c in default_candidates(cfg) if c.name == name)
        ok, why = cand.available()
        if not ok:
            print(f"{label:<13} UNAVAILABLE: {why}")
            rows.append({"backend": label, "status": "unavailable", "reason": why})
            continue
        try:
            t0 = time.perf_counter()
            backend = cand.factory(cfg)
            init_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<13} FAILED TO LOAD: {exc}")
            rows.append({"backend": label, "status": "failed", "reason": str(exc)})
            failures += 1
            continue
        out, t = bench(backend, crops)
        r = compare(ref, out)
        tol = TOL[backend.precision]
        passed = r["finite"] and r["min_cosine"] >= tol and r["nearest_neighbour_agreement"] == 1.0
        failures += not passed
        rows.append({"backend": label, "device": backend.device, "precision": backend.precision,
                     "init_s": round(init_s, 3), "ms_per_crop": 1000 * t / len(crops),
                     "tolerance_min_cosine": tol, "passed": passed, **r})
        print(f"{label:<13} [{backend.device}] {1000 * t / len(crops):7.2f} ms/crop  "
              f"min cos={r['min_cosine']:.7f} max|diff|={r['max_abs_diff']:.2e} "
              f"max dSim={r['max_pairwise_similarity_change']:.2e} "
              f"NN agree={r['nearest_neighbour_agreement']:.3f} {'OK' if passed else 'MISMATCH'}")
        backend.close()
    if args.json:
        args.json.write_text(json.dumps({"crops": len(crops), "order": ORDER, "rows": rows},
                                        indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
