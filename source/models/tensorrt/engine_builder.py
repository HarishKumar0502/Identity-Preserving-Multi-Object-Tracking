"""Reproducible ONNX -> TensorRT engine build with metadata and staleness checks.

Artifacts (never overwrite the source ONNX / PyTorch checkpoint)::

    source/models/tensorrt/reid_resnet18_fp32.engine        serialized engine
    source/models/tensorrt/reid_resnet18_fp32.engine.json   build metadata (below)
    source/models/tensorrt/reid_resnet18_fp16.onnx          derived FP16-typed graph (FP16 only)

An engine is only reused when its metadata matches the current ONNX file (SHA-256),
TensorRT version, GPU name / compute capability, precision, input size and batch
profile; otherwise it is reported as stale and rebuilt. TensorRT 11 networks are
strongly typed (there is no FP16 builder flag), so FP16 is built from an FP16-typed
copy of the graph with FP32 input/output casts, keeping the backend interface FP32.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from source.utils.device_utils import ensure_gpu_runtime_on_path, probe_cuda

log = logging.getLogger("ps2.tensorrt")


class EngineBuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_path_for(tensorrt_dir: Path, precision: str) -> Path:
    return Path(tensorrt_dir) / f"reid_resnet18_{precision}.engine"


def metadata_path(engine_path: Path) -> Path:
    return Path(str(engine_path) + ".json")


def make_fp16_onnx(src: Path, dst: Path) -> Path:
    """FP16-typed copy of ``src``: FP16 weights/compute, FP32 graph input and output."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    g = model.graph
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16),
                                                  init.name))
    for node in g.node:  # attribute tensors (Constant nodes) too
        for attr in node.attribute:
            if attr.t is not None and attr.t.data_type == TensorProto.FLOAT:
                attr.t.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(attr.t).astype(np.float16), attr.t.name))
    in_name, out_name = g.input[0].name, g.output[0].name
    for node in g.node:
        node.input[:] = [f"{in_name}__fp16" if i == in_name else i for i in node.input]
        node.output[:] = [f"{out_name}__fp16" if o == out_name else o for o in node.output]
    g.node.insert(0, helper.make_node("Cast", [in_name], [f"{in_name}__fp16"],
                                      to=TensorProto.FLOAT16, name="input_to_fp16"))
    g.node.append(helper.make_node("Cast", [f"{out_name}__fp16"], [out_name],
                                   to=TensorProto.FLOAT, name="output_to_fp32"))
    del g.value_info[:]
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return dst


def expected_metadata(onnx_path: Path, precision: str, input_hw: Tuple[int, int],
                      max_batch: int, opt_batch: int) -> dict:
    ensure_gpu_runtime_on_path()
    import tensorrt as trt

    gpu = probe_cuda()
    return {"tensorrt_version": trt.__version__, "precision": precision,
            "onnx_file": Path(onnx_path).name, "onnx_sha256": sha256(onnx_path),
            "input_hw": list(input_hw), "min_batch": 1, "opt_batch": opt_batch,
            "max_batch": max_batch, "gpu_name": gpu.get("name"),
            "compute_capability": gpu.get("compute_capability")}


_COMPARED = ("tensorrt_version", "precision", "onnx_sha256", "input_hw", "max_batch",
             "gpu_name", "compute_capability")


def engine_status(engine_path: Path, onnx_path: Path, precision: str, input_hw, max_batch: int,
                  opt_batch: int) -> Tuple[bool, str]:
    """(usable, reason). Usable only if engine + metadata exist and match this setup."""
    engine_path = Path(engine_path)
    if not engine_path.is_file():
        return False, "engine file not found"
    meta_file = metadata_path(engine_path)
    if not meta_file.is_file():
        return False, "engine metadata missing (unknown provenance)"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"engine metadata unreadable: {exc}"
    want = expected_metadata(onnx_path, precision, input_hw, max_batch, opt_batch)
    diffs = [f"{k}: engine={meta.get(k)!r} current={want[k]!r}" for k in _COMPARED
             if meta.get(k) != want[k]]
    if diffs:
        return False, "engine is stale (" + "; ".join(diffs) + ")"
    if meta.get("engine_sha256") and meta["engine_sha256"] != sha256(engine_path):
        return False, "engine file does not match its metadata checksum (corrupted?)"
    return True, "ok"


def build_engine(onnx_path: Path, engine_path: Path, precision: str = "fp32",
                 input_hw: Tuple[int, int] = (256, 128), max_batch: int = 32,
                 opt_batch: int = 8, workspace_mb: int = 1024) -> dict:
    """Parse ONNX, build a dynamic-batch engine, write engine + metadata atomically."""
    ensure_gpu_runtime_on_path()
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise EngineBuildError("TensorRT Python package is not installed.") from exc
    gpu = probe_cuda()
    if not gpu["available"]:
        raise EngineBuildError(f"No usable CUDA device: {gpu['reason']}")
    if precision not in ("fp32", "fp16"):
        raise EngineBuildError(f"Unsupported precision '{precision}'.")
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    if not onnx_path.is_file():
        raise EngineBuildError(f"ONNX model not found: {onnx_path.name}")
    graph_path = onnx_path
    if precision == "fp16":
        graph_path = make_fp16_onnx(onnx_path, engine_path.with_suffix(".onnx"))

    t0 = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    elif hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # TensorRT 8.x
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(graph_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EngineBuildError(f"TensorRT could not parse the ONNX model (unsupported operator "
                               f"or opset?): {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)  # strict FP32 reference precision
    if precision == "fp16" and not flags and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)  # weakly typed TensorRT < 10.12
    h, w = input_hw
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, 3, h, w), (min(opt_batch, max_batch), 3, h, w),
                      (max_batch, 3, h, w))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise EngineBuildError("TensorRT engine build failed (see TensorRT log).")
    build_s = time.perf_counter() - t0

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(bytes(serialized))
    os.replace(tmp, engine_path)
    meta = expected_metadata(onnx_path, precision, input_hw, max_batch, opt_batch)
    meta.update(engine_file=engine_path.name, engine_sha256=sha256(engine_path),
                engine_bytes=engine_path.stat().st_size, graph_file=graph_path.name,
                input_name=inp.name, output_name=network.get_output(0).name,
                driver_cuda=gpu.get("driver_cuda"), build_time_s=round(build_s, 2),
                built_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    metadata_path(engine_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("TensorRT engine built: %s (%s, %.1fs)", engine_path.name, precision, build_s)
    return meta


def ensure_engine(onnx_path: Path, engine_path: Path, precision: str, input_hw, max_batch: int,
                  opt_batch: int, auto_build: bool) -> Tuple[Path, Optional[dict]]:
    """Return a usable engine path, building it when missing/stale and ``auto_build``."""
    ok, reason = engine_status(engine_path, onnx_path, precision, input_hw, max_batch, opt_batch)
    if ok:
        return Path(engine_path), None
    if not auto_build:
        raise EngineBuildError(f"TensorRT engine not usable: {reason}. Run "
                               "scripts/build_reid_tensorrt.py.")
    log.warning("TensorRT engine not usable (%s); building it now (one-off, cached).", reason)
    meta = build_engine(onnx_path, engine_path, precision, tuple(input_hw), max_batch, opt_batch)
    return Path(engine_path), meta
