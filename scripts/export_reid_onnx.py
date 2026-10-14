"""Export the PS-1 ReID model (embedding head) to ONNX and validate against PyTorch.

    python scripts/export_reid_onnx.py [--out source/models/onnx/reid_resnet18.onnx] [--opset 17]

The model is rebuilt and strict-loaded from the checkpoint first; export never runs
on an unvalidated state dict. Input: float32 (B, 3, H, W) ImageNet-normalized RGB,
dynamic batch. Output: "embedding" (B, 512) raw pooled features (backends L2-normalize).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from source.config.settings import REID  # noqa: E402
from source.models.reid.inference import build_reid_model, l2_normalize  # noqa: E402


def export(out: Path, opset: int = 17) -> Path:
    import torch

    model = build_reid_model(REID).cpu().eval()
    h, w = REID.input_size
    dummy = torch.randn(2, 3, h, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["embedding"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                  do_constant_folding=True)
    try:
        # Legacy TorchScript exporter: no onnxscript dependency, stable for ResNets.
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kwargs)
    except TypeError:  # very old torch without the `dynamo` flag
        torch.onnx.export(model, (dummy,), str(out), **kwargs)
    return out


def validate(onnx_path: Path, batch_sizes=(1, 4, 16), tol: float = 1e-4) -> dict:
    import onnx
    import onnxruntime as ort
    import torch

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    model = build_reid_model(REID).cpu().eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    h, w = REID.input_size
    rng = np.random.default_rng(0)
    report = {}
    for b in batch_sizes:
        x = rng.normal(size=(b, 3, h, w)).astype(np.float32)
        with torch.inference_mode():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        if got.shape != ref.shape:
            raise RuntimeError(f"Shape mismatch at batch {b}: {got.shape} vs {ref.shape}")
        max_abs = float(np.abs(got - ref).max())
        cos = float((l2_normalize(got) * l2_normalize(ref)).sum(1).min())
        report[b] = {"shape": list(got.shape), "max_abs_diff": max_abs, "min_cosine": cos}
        if max_abs > tol * max(1.0, float(np.abs(ref).max())) or cos < 0.9999:
            raise RuntimeError(f"ONNX output deviates from PyTorch at batch {b}: {report[b]}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REID.onnx_path)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    path = export(args.out, args.opset)
    print(f"exported: {path} ({path.stat().st_size / 2**20:.1f} MB)")
    try:
        report = validate(path)
    except ImportError as exc:
        print(f"validation skipped: {exc} (install onnx and onnxruntime)")
        return
    for b, r in report.items():
        print(f"batch {b:>2}: shape={r['shape']} max|diff|={r['max_abs_diff']:.2e} "
              f"min cos={r['min_cosine']:.6f}")
    print("ONNX validation PASSED")


if __name__ == "__main__":
    main()
