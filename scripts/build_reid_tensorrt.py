"""Build the TensorRT engine for the ReID ONNX model, then validate it vs PyTorch.

    python scripts/build_reid_tensorrt.py [--precision fp32|fp16] [--max-batch 32] [--force]

Requirements: NVIDIA GPU + driver, the TensorRT Python package and ``cuda-python``
(see README "TensorRT"). No CUDA build of torch is needed. Outputs (the ONNX model and
the PyTorch checkpoint are never modified)::

    source/models/tensorrt/reid_resnet18_<precision>.engine(.json)

The engine is specific to the GPU model, TensorRT version and driver stack it was
built with: the metadata file records them and the app rebuilds stale engines.
Exit codes: 0 built + validated, 2 environment not capable, 1 build/validation failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from source.config.settings import REID  # noqa: E402
from source.utils.device_utils import probe_cuda, tensorrt_version  # noqa: E402

MIN_COS = {"fp32": 0.99999, "fp16": 0.999}


def check_environment() -> list:
    problems = []
    v = tensorrt_version()
    print(f"TensorRT: {v or 'not installed'}")
    if v is None:
        problems.append("TensorRT Python package not installed (pip install tensorrt-cu12 "
                        "cuda-python, see README).")
    elif tuple(int(p) for p in v.split(".")[:2]) < (8, 6):
        problems.append(f"TensorRT {v} < 8.6 (tensor-name API required).")
    gpu = probe_cuda()
    print(f"CUDA device: {gpu}")
    if not gpu["available"]:
        problems.append(f"No usable CUDA GPU: {gpu['reason']}")
    if not REID.onnx_path.is_file():
        problems.append("ONNX model missing: run scripts/export_reid_onnx.py first.")
    return problems


def validate(cfg, engine_path: Path) -> dict:
    from source.logic.inference_manager import probe_crops
    from source.models.model_registry import create_backend
    from source.models.tensorrt.inference import TensorRTBackend

    trt_b = TensorRTBackend(cfg, engine_path)
    ref_b = create_backend("pytorch-cpu", cfg)
    rng = np.random.default_rng(0)
    h, w = cfg.input_size
    report, worst = {}, 1.0
    inputs = {f"random_b{b}": rng.normal(size=(b, 3, h, w)).astype(np.float32) for b in (1, 4, 16, 32)}
    from source.models.reid.inference import preprocess_crops
    inputs["probe_crops"] = preprocess_crops(probe_crops(8), cfg)
    for name, x in inputs.items():
        ref = ref_b.embed_preprocessed(x)
        got = trt_b.embed_preprocessed(x)
        cos = float((ref * got).sum(1).min())
        worst = min(worst, cos)
        report[name] = {"batch": len(x), "min_cosine": cos,
                        "max_abs_diff": float(np.abs(ref - got).max())}
        print(f"{name:>12}: min cosine vs PyTorch = {cos:.7f}  max|diff| = "
              f"{report[name]['max_abs_diff']:.2e}")
    trt_b.close()
    if worst < MIN_COS[cfg.tensorrt_precision]:
        raise RuntimeError(f"TensorRT embeddings deviate (min cos {worst:.6f} < "
                           f"{MIN_COS[cfg.tensorrt_precision]}).")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16"], default=REID.tensorrt_precision)
    ap.add_argument("--fp16", action="store_true", help="alias for --precision fp16")
    ap.add_argument("--max-batch", type=int, default=REID.tensorrt_max_batch)
    ap.add_argument("--force", action="store_true", help="rebuild even if the engine is current")
    args = ap.parse_args()
    precision = "fp16" if args.fp16 else args.precision
    problems = check_environment()
    if problems:
        print("TensorRT engine NOT built:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    from source.models.tensorrt.engine_builder import build_engine, engine_status, metadata_path

    cfg = replace(REID, tensorrt_precision=precision, tensorrt_max_batch=args.max_batch)
    path = cfg.tensorrt_path
    ok, reason = engine_status(path, cfg.onnx_path, precision, cfg.input_size, args.max_batch,
                               cfg.tensorrt_opt_batch)
    if ok and not args.force:
        print(f"engine is current: {path}")
    else:
        print(f"building engine ({reason if not ok else 'forced'}) ...")
        try:
            meta = build_engine(cfg.onnx_path, path, precision, cfg.input_size, args.max_batch,
                                cfg.tensorrt_opt_batch)
        except Exception as exc:  # noqa: BLE001
            print(f"BUILD FAILED: {exc}")
            sys.exit(1)
        print(f"engine written: {path} ({precision}, max batch {args.max_batch}, "
              f"{meta['build_time_s']} s)")
    try:
        report = validate(cfg, path)
    except Exception as exc:  # noqa: BLE001
        print(f"VALIDATION FAILED: {exc}")
        sys.exit(1)
    meta_file = metadata_path(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["validation_vs_pytorch_cpu"] = report
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("TensorRT validation PASSED")


if __name__ == "__main__":
    main()
