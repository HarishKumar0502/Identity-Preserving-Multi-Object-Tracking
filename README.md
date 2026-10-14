# PS-2: Identity-Preserving Multi-Object Tracking Through Dense Occlusion

An online (causal) person tracker: YOLO person detector + the **PS-1 ResNet-18 ReID model** (reused, not retrained) + Kalman filter + gated Hungarian association + an explicit track lifecycle with conservative identity recovery. It is served through FastAPI. Swagger UI at `/docs` is the only interface. **The user only uploads a video**: the inference backend (TensorRT / ONNX Runtime / PyTorch, GPU or CPU) and all tracking parameters are chosen internally.

## 1. Environment (verified 2026-10-13)

| Item | Value |
|---|---|
| OS / Python | Windows 11, Python 3.14.7 (`C:\Python314\python.exe`), no virtual environment |
| GPU / driver | NVIDIA GeForce RTX 2050 (4 GB, Ampere sm_86), driver 555.97 (CUDA 12.5 driver API) |
| CUDA toolkit / cuDNN | not installed (not needed: TensorRT ships its own libraries, buffers use the driver API) |
| torch / torchvision | 2.14.1 **+cpu** / 0.29.1 (CPU build; unchanged) |
| onnx / onnxruntime | 1.23.2 / 1.31.0 (providers: CPU, Azure; no CUDA/DirectML EP) |
| **TensorRT** | **11.4.0.106** (`tensorrt-cu12`) + `cuda-python` 12.9.7, in `.gpu_runtime/` (see §5.3) |
| opencv-python | 5.0.0.93 (note: `opencv-python-headless` 5.0.0.93 is *also* installed; both provide `cv2`. Keep them at the same version or uninstall one) |
| ultralytics / fastapi | 8.4.175 / 0.136.0 |

`python -m pip install -r requirements.txt` installs the CPU stack. GPU acceleration is optional and auto-detected (§5).

## 2. Model placement

All model files live under `source/models/`, one folder per model, each next to its loading/inference code (`inference.py`):

| File | Purpose | Notes |
|---|---|---|
| `source/models/reid/best.pth` | PS-1 ReID checkpoint (PyTorch backend + shared preprocessing: `reid/inference.py`) | lookup: `$PS2_REID_CHECKPOINT` → `source/models/reid/best.pth` |
| `source/models/detector/yolo11n.pt` | COCO detector, class 0 = person (`detector/inference.py`) | `$PS2_DETECTOR_WEIGHTS` |
| `source/models/onnx/reid_resnet18.onnx` | ReID exported to ONNX (opset 17, input `input [batch,3,256,128]`, output `embedding [batch,512]`); ONNX Runtime backend in `onnx/inference.py` | `scripts/export_reid_onnx.py` |
| `source/models/tensorrt/reid_resnet18_fp32.engine` (+ `.json`) | TensorRT engine + build metadata (generated per machine); runner in `tensorrt/inference.py`, builder in `tensorrt/engine_builder.py` | auto-built at start-up, or `scripts/build_reid_tensorrt.py` |
| `source/models/tensorrt/reid_resnet18_fp16.engine` (+ `.json`, `.onnx`) | optional FP16 engine and the derived FP16-typed graph | `scripts/build_reid_tensorrt.py --precision fp16` |

The source ONNX model and the PyTorch checkpoint are never modified by any build step. `source/models/model_registry.py` maps backend names to these implementations and caches the loaded detector and ReID backend per process.

### What the PS-1 checkpoint actually is (inspected)

`best.pth` is a dict: `epoch=11`, `val_tar_at_far=0.855`, `experiment={name: D_cosface_triplet_hard, head: cosface, s=30, m=0.25, triplet batch-hard, P=16, K=4, ...}` and `model_state_dict`. The state dict is an exact **torchvision `resnet18`** with `fc: 751×512`. Its rows are unit-norm and its bias is zero, i.e. the CosFace head.

Loading (`source/models/reid/inference.py::build_reid_model`): rebuild `torchvision.models.resnet18(num_classes=751)` → **`load_state_dict(strict=True)`** → replace `fc` with `Identity` → `eval()`. The embedding is the 512-d global-average-pooled feature, L2-normalized, compared by cosine distance.

**Preprocessing assumption:** The PS-1 training code was not provided, so the input size can't be read from the checkpoint. I use the Market-1501 standard: BGR→RGB, resize to **256×128 (H×W)**, ImageNet mean/std (`PS2_REID_INPUT_H/W` to override). Preprocessing is shared by every backend.

## 3. Run

```powershell
python "D:\zed digital\ps2\source\main.py"   # from any directory; 127.0.0.1:8000, PS2_HOST / PS2_PORT to change
# equivalent (from the project root):
python -m uvicorn source.main:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

**Use one of these two commands on Windows.** Plain `python -m uvicorn source.main:app` runs on asyncio's *Proactor* loop, and that loop has two problems when a client aborts a connection (a cancelled video download, a closed tab, a browser preconnect socket):

* It logs `ConnectionResetError: [WinError 10054] ... _ProactorBasePipeTransport._call_connection_lost()`. This is noise; the request itself is fine.
* If the reset arrives while the connection is being accepted, it logs `Accept failed on a socket ... [WinError 64]` and **closes the listening socket**. Every later request is then refused until a restart.

This was reproduced here: the server stopped accepting after 33 RST-aborted connections. With the Selector loop it survived 300 aborted connections plus a full job, with zero errors (`tests/test_server_loop.py`). If the server is still started the old way, it logs a warning at start-up and suppresses the benign 10054 traceback. It cannot prevent the accept failure, so use the commands above.

Open **http://127.0.0.1:8000/docs**. At start-up the server removes stale files from earlier runs, loads the detector and **selects and verifies the ReID backend** (first start on a new GPU machine also builds the TensorRT engine, ~17 s; later starts load it in < 1 s). The log line `ReID backend selected: ...` and `GET /health` show the choice.

1. **POST /track** → *Try it out* → choose a video file → *Execute*. That is the only input. Returns `202` with `job_id`, video metadata and the ReID backend that will be used.
2. **GET /jobs/{job_id}**: `queued|processing|completed|failed`, `progress_percent`, frame counter, `reid_backend` (actual backend/device/precision), `processing_time_s`, `error` (actionable message), and when completed a `result` block (frames, unique IDs, FPS, `warnings`, links).
3. **GET /jobs/{job_id}/video**: annotated video (H.264 MP4 via Windows Media Foundation, browser-playable; fallbacks in §5.5). Solid box = observed; dashed box = predicted position while occluded; `[recovered]` = identity restored after loss.
4. **GET /jobs/{job_id}/tracks**: tracks in MOTChallenge format (`frame,id,x,y,w,h,score,-1,-1,-1`).
5. **GET /jobs/{job_id}/summary**: frames, FPS, unique IDs, tracks created, recoveries, occlusion events, per-stage timing, latency, peak RSS, ReID backend selection report, tracker config. `id_switches` is `null` (not measurable without ground truth).
6. **GET /health**: detector/ReID status, selection attempts with reasons, GPU/TensorRT detection, artifacts, codec candidates.
`/docs` is FastAPI's Swagger UI with one change (`source/api/docs.py`): file responses (`/video`, `/tracks`) are not buffered in the page; *Download file* links to the real URL so the browser's download manager saves it. Stock Swagger buffers files in a browser Blob, which Chrome refuses when the system drive is almost full and then shows `Error: OK`.
7. **GET /ready**: readiness probe, `200` once the detector and a verified ReID backend are loaded, else `503` with the load errors.

The whole video is always processed. Previous versions exposed `detection_confidence`, `track_buffer`, `reassociation_threshold`, `tracker_mode`, `reid_backend` and `max_frames` on `/track`; they were removed (old clients that still send them are not rejected, the fields are ignored). Internal defaults are in `source/config/settings.py`; advanced overrides are environment variables, or a `.env` file at the project root (template: `.env.example`; real environment variables win) (`PS2_REID_BACKEND`, `PS2_TENSORRT_PRECISION`, `PS2_VIDEO_CODECS`, …) or the CLI below.

Limits: 500 MB upload (streamed to disk in 1 MB chunks; multipart spooling goes to `uploads/.tmp`, not `%TEMP%`), extensions `.mp4 .avi .mov .mkv .m4v .webm`, decodability checked before queueing (≥ 2 decodable frames, so a renamed still image is rejected), 1 concurrent job and up to 8 pending (`PS2_MAX_CONCURRENT_JOBS`, `PS2_MAX_QUEUED_JOBS`). The upload is deleted when its job ends; a failed job's partial outputs are deleted; finished jobs older than 24 h are removed.

`python worker.py` with no arguments starts this same backend server. Without the API (root-level worker, same pipeline): `python worker.py video.mp4 --out outputs\cli [--backend auto|tensorrt|onnx-cpu|pytorch-cpu] [--mode baseline] [--max-frames N]`. It validates the input before loading models, writes `tracked.mp4`, `tracks_mot.txt`, `summary.json` and `job.json` (status/progress/error), removes partial results on failure, and exits `0` (completed), `1` (job failed) or `2` (invalid arguments/input).

End-to-end check against a running server (upload → poll → download → decode): `python scripts\e2e_api_test.py video1.mp4 video2.mp4 [--concurrent] --url http://127.0.0.1:8000`

## 4. Tracking algorithm (`source/logic/tracking/track_manager.py`)

Per frame (online, causal):

1. **Kalman predict** for every track. State `(cx, cy, aspect, h, velocities)`; noise scales with box height, so covariance grows while a track is unobserved. Measurement noise is inflated for low-confidence detections.
2. **Stage 1** (confirmed/occluded tracks × detections ≥ `high_conf`), fused cost with each term in [0, 1]:
   `cost = (w_m·min(maha/χ²₀.₉₅,1) + w_iou·(1−IoU) + w_app·d_cos) / Σw`, weights 0.2 / 0.3 / 0.5.
   - Motion gate: Mahalanobis > 9.49 (χ², 4 DoF) is rejected unless IoU ≥ 0.3.
   - Appearance gate: d_cos > 0.45 is rejected when IoU < 0.5. With strong overlap (IoU ≥ 0.5), only d_cos > 0.9 vetoes, and the motion term is zeroed because overlap already validates the motion.
   - Partial or overlapped crops get half the appearance weight and a looser gate.
   - With no appearance evidence, IoU ≥ 0.1 is required.
   - Hungarian solves; each pair is then re-checked against `match_thresh`, so gated pairs can never be accepted.
3. **Stage 2** (remaining active tracks × low-score boxes): IoU only. These are usually occluded people. They keep tracks alive but never update appearance memory.
4. **Stage 3** (tentative tracks × remaining confident boxes): IoU plus the appearance gate.
5. **Stage 4: re-association** (lost tracks × remaining confident boxes). A match must pass all of:
   - a spatial window that grows with time lost (1.5 → max 4 track heights);
   - a plausible size ratio;
   - long-term gallery distance ≤ `reid_thresh`;
   - only **reliable** crops are allowed;
   - a **ratio test** against the second-best candidate. Ambiguous matches are rejected and counted.
6. **New tracks** from leftover boxes with score ≥ 0.6. A box with IoU ≥ 0.8 against a track updated this frame is suppressed as a duplicate. On **confirmation** (3 hits), a new track gets one more recovery attempt using the *mean of its multi-frame features*. Lookalikes therefore need several observations before an old ID is reused.
7. **Lifecycle**: `tentative → confirmed → occluded` (missed ≤ 15 frames, still predicted) `→ lost` (velocity frozen; kept for `track_buffer` frames; at most 200 kept) `→ expired`. A recovered track is `reassociated` for that frame, then becomes `confirmed` again. Display IDs are issued only on confirmation, so tentative noise doesn't burn IDs. They are unique per video, and recovery restores the original ID.

**Two appearance memories.**
- *Long-term gallery* (≤ 30 features + EMA): only crops with score ≥ 0.6, < 30 % covered by another box, not cut off by the frame border, not an outlier against the track's identity, and from unambiguous matches. This is the only memory used to restore an identity.
- *Short-term*: the latest valid embedding of any quality, used only for frame-to-frame matching of active tracks.

This split came out of debugging on the sample video. People entering at the frame edge would otherwise have no appearance memory and fragment; mixing partial crops into the gallery, on the other hand, produced a false identity recovery.

All thresholds live in `source/config/settings.py::TrackerConfig`. `baseline_tracker_config()` is the IoU + Kalman, short-memory, no-ReID variant.

## 5. Inference backends and automatic selection

### 5.1 Common interface

`backend.extract_embeddings(list_of_BGR_crops) -> (N, 512)` L2-normalized float32. Preprocessing (resize 256×128, RGB, ImageNet normalization), embedding dimension and normalization are identical for every backend; the tracker never sees which one is active.

### 5.2 Selection order (`source/logic/inference_manager.py`)

At start-up (`PS2_REID_BACKEND=auto`, the default) the candidates are tried in this order. The first one that **loads the real model and passes verification** is used and cached for all jobs:

| # | Candidate | Requirements checked before loading |
|---|---|---|
| 1 | `tensorrt` – native TensorRT engine | `tensorrt` importable, CUDA device via driver API, ONNX model present (engine auto-built/rebuilt if missing or stale) |
| 2 | `onnx-cuda` – ONNX Runtime CUDA EP | `CUDAExecutionProvider` in `onnxruntime` |
| 3 | `onnx-dml` – ONNX Runtime DirectML | `DmlExecutionProvider` in `onnxruntime` |
| 4 | `pytorch-cuda` | `torch.cuda.is_available()` |
| 5 | `onnx-cpu` | `onnxruntime` + ONNX model (faster than PyTorch CPU here) |
| 6 | `pytorch-cpu` | torch + checkpoint (the reference implementation) |

**Verification** of every candidate: a fixed, deterministic probe batch is embedded at batch 1 and batch 4. The result must have shape `(B, 512)`, be finite and unit-norm, be batch-consistent, and agree with the PyTorch-CPU reference to cosine ≥ `0.9995` (`PS2_REID_VERIFY_MIN_COSINE`). A backend that loads but computes something different is rejected. A CUDA GPU alone never implies TensorRT: each step is checked.

**Logging/reporting:** every attempt is logged with its reason (`unavailable` / `failed` / `selected`), plus init time, warm-up time, device, precision, model file and agreement with the reference. If a CPU backend is selected although GPU candidates exist, a WARNING lists why each GPU backend was unusable. `/health → reid.selection`, every job's `reid_backend` and the summary's `reid_backend_selection` expose the same report.

**Runtime fallback:** the selected backend is wrapped in `ResilientBackend`. If inference raises mid-video (CUDA error, OOM, driver reset), the next candidate is loaded, **verified the same way**, and the failed batch is retried. The job continues with identical embeddings (within tolerance) and records a `fallback_events` entry and a job warning.

Advanced/internal override: `PS2_REID_BACKEND=tensorrt|onnx-cuda|onnx-dml|pytorch-cuda|onnx-cpu|pytorch-cpu` (or legacy `onnx` / `pytorch`) restricts the candidates. The API never takes this from the client.

### 5.3 TensorRT (native engine, not the ONNX Runtime TensorRT EP)

Native TensorRT was chosen over ONNX Runtime's TensorRT Execution Provider because the project already had a native runner. The EP would also require replacing `onnxruntime` with `onnxruntime-gpu` plus several GB of CUDA/cuDNN wheels. The runner (`TensorRTBackend`) allocates device buffers with the CUDA **driver** API (`cuda-python` → `nvcuda.dll`, part of the display driver). A CUDA build of torch is therefore **not** needed, and the CPU torch install was left untouched.

**Original error.** `ModuleNotFoundError: No module named 'tensorrt'` → `ReIDModelError: TensorRT Python package is not installed`. TensorRT was simply not installed. Even with it installed, the old runner required a CUDA build of torch, which this machine does not have. Jobs with `reid_backend='tensorrt'` therefore always failed. They also reported only "Internal processing error (ReIDModelError)".

**Installation used here (reproducible):**

```powershell
# Python 3.14 bindings exist only for TensorRT >= 11.1 (10.x has no cp314 wheels).
# The libs wheel unpacks to 3.2 GB; C: had 2.7 GB free, so it is installed with --target
# into the project-local .gpu_runtime/ (git-ignored). The app appends that folder to
# sys.path only if it exists (PS2_GPU_RUNTIME overrides the location). No venv, and no
# existing package is modified.
$env:TMP="D:\ps2_wheels\tmp"; $env:TEMP=$env:TMP
python -m pip download --no-cache-dir -d D:\ps2_wheels\dl tensorrt-cu12==11.4.0.106 cuda-python==12.9.7
python -m pip download --no-cache-dir --no-deps -d D:\ps2_wheels\dl --index-url https://pypi.nvidia.com/ tensorrt_cu12_libs==11.4.0.106
python -m pip install --no-cache-dir --find-links D:\ps2_wheels\dl --target .gpu_runtime tensorrt-cu12==11.4.0.106 cuda-python==12.9.7
# With enough space on the system drive the plain form is fine:
#   python -m pip install tensorrt-cu12==11.4.0.106 cuda-python==12.9.7
```

**Compatibility check (performed):**

* ONNX model: opset 17, IR 8, ops `Conv ×20, Relu ×17, Add ×8, MaxPool, GlobalAveragePool, Flatten`. Input `input [batch,3,256,128]` float32, output `embedding [batch,512]`, dynamic batch.
* The TensorRT 11.4 ONNX parser reports zero errors.
* The GPU is sm_86, and the driver (CUDA 12.5) runs the CUDA 12.x TensorRT build.

**Build** (`source/models/tensorrt/engine_builder.py`, CLI `scripts/build_reid_tensorrt.py`):

* The network is strongly typed. The optimization profile is batch 1 / 8 / 32 (min/opt/max) with a 1 GB workspace.
* **FP32** is built with TF32 disabled (strict FP32).
* **FP16**: TensorRT 11 has no FP16 builder flag. FP16 is built from a derived FP16-typed copy of the graph (`*_fp16.onnx`, FP32 input/output casts).
* The engine is written atomically, with a `.json` metadata file: TensorRT version, GPU name/compute capability, driver, ONNX SHA-256, precision, input size, batch profile, engine SHA-256, build time, and the validation report.
* At load time the metadata must match the current ONNX file, TensorRT version, GPU and settings, and the engine checksum must match. Otherwise the engine is reported **stale/corrupted** and rebuilt (auto-build is on by default, `PS2_TENSORRT_AUTO_BUILD=0` disables it).
* Missing package, no GPU, parse failure (unsupported operator), build failure, deserialization failure (corrupted / other GPU / other TensorRT version) and execution-context failure each raise an explicit `ReIDModelError`. Selection then moves on to the next backend.

```powershell
python scripts\build_reid_tensorrt.py                    # FP32 (default); builds if missing/stale, validates
python scripts\build_reid_tensorrt.py --precision fp16   # optional FP16
python scripts\validate_backends.py --video samples\people-detection.mp4 --fp16
```

**Verified on this machine (actual runs):**

| Engine | Build time | Size | vs PyTorch-CPU (random batches 1/4/16/32 + crops) |
|---|---|---|---|
| FP32 | 16.9 s | 46.9 MB | min cosine 0.9999998, max \|diff\| 2.0e-7 |
| FP16 | 15.8 s | 22.7 MB | min cosine 0.9999954, max \|diff\| 7.3e-4 |

FP32 is the default. FP16 passes the accuracy check but is opt-in (`PS2_TENSORRT_PRECISION=fp16`); see §5.4 for its measured speed. Engines are tied to the GPU model, TensorRT version and driver stack. Rebuild on each target (the app does this automatically) rather than copying engines. On a Jetson, use the JetPack TensorRT and the same script.

### 5.4 Performance (same video, same machine)

Measured 2026-10-13 on this laptop: Intel CPU (Family 6 Model 154), RTX 2050 4 GB, Windows 11, TensorRT 11.4. Video: **MOT17-09** raw clip, 525 frames, 960×540 @ 30 FPS = 17.5 s. Measured with offline benchmark tooling that is no longer part of this backend repository. Raw JSON: `D:\ps2_testdata\logs\`.

**A. ReID model inference only** (pre-processed batch, median of 5 interleaved rounds, so all backends saw the same CPU state; shared CPU pre-processing costs 0.64 ms/crop on top):

| Backend | Device | b1 ms/crop | b8 ms/crop | b32 ms/crop | vs PyTorch-CPU (min cosine, 64 real crops) |
|---|---|---|---|---|---|
| tensorrt FP32 | RTX 2050 | 4.81 | **0.82** | 0.82 | 0.9999999 |
| tensorrt FP16 | RTX 2050 | 1.15 | **0.30** | 0.27 | 0.9999932 |
| onnx-cpu | CPU | 5.39 | 4.41 | 4.51 | 0.9999998 |
| pytorch-cpu (reference) | CPU | 16.37 | 11.18 | 11.95 | 1 |

At batch ≥ 8, TensorRT FP32 is **5.4× faster than ONNX-CPU and 13.6× faster than PyTorch-CPU** for ReID inference; FP16 is a further 2.7×. At batch 1 the GPU's transfer/launch overhead dominates and FP32 TensorRT is barely faster than ONNX-CPU.

**B. Full pipeline** (decode → YOLO detection *on CPU* → ReID → association → render → H.264 encode → output validation). The machine throttled after the first two runs: the CPU-bound `detect` stage rose from 15.6 s to ~38 s for *every* backend. Only runs with similar `detect` time are comparable, so they are grouped:

| Backend | Run | Init + verify s | Warm-up s | Detect s | ReID stage s* | Pipeline s | FPS | Real-time factor | GPU peak MB† | IDs |
|---|---|---|---|---|---|---|---|---|---|---|
| *CPU at full clock* | | | | | | | | | | |
| tensorrt FP32 | 1 | 3.25 | 0.06 | 15.6 | 11.9 | **31.8** | **16.9** | 0.56 | 252 | 24 |
| tensorrt FP16 | 1 | 3.18 | 0.03 | 15.9 | 11.3 | 31.4 | 17.1 | 0.57 | 174 | 24 |
| *CPU throttled (comparable group)* | | | | | | | | | | |
| tensorrt FP32 | 2 | 4.57 | – | 38.2 | 26.5 | 74.3 | 7.2 | 0.24 | 252 | 24 |
| tensorrt FP16 | 2 | 5.19 | – | 38.9 | 18.5 | 67.2 | 8.0 | 0.27 | 174 | 24 |
| onnx-cpu | 1 | 3.17 | – | 39.6 | 20.0 | 67.8 | 7.8 | 0.26 | 0 | 24 |
| pytorch-cpu | 1 | 3.07 | – | 37.2 | 71.3 | 117.2 | 4.5 | 0.15 | 0 | 24 |
| pytorch-cpu | 2 | 4.89 | – | 39.2 | 72.9 | 123.0 | 4.3 | 0.14 | 0 | 24 |
| *disturbed (excluded)* onnx-cpu | 2 | 4.23 | – | 416.3 | 66.3 | 502.3 | 1.0 | – | 0 | 24 |

\* The `reid` stage includes CPU crop extraction, pre-processing and overlap analysis, not only model inference (see A).
† Device-wide NVML used memory above the idle baseline, measured while the run was active.
"Init + verify" covers loading the cached model/engine plus the reference check, which loads PyTorch-CPU once (~2.5 s). Engine *build* time (first start only) is 16.9 s FP32 / 15.8 s FP16.

Conclusions supported by these measurements:

* TensorRT makes ReID inference 5–14× faster and keeps embeddings identical within 1e-7 (FP32). All backends produced the same 24 IDs and 3632 track rows.
* End to end, TensorRT is clearly faster than PyTorch-CPU, the previous default (≈ 1.6× under equal conditions). It is **not** clearly faster than ONNX-CPU, because the pipeline is bound by the **CPU YOLO detector** (≈ 50–55 % of the time) and CPU pre-processing.
* No configuration reached real time on this laptop: at best 16.9 FPS on a 30 FPS video.
* The next speed-up must come from running the detector on the GPU. That needs either a CUDA build of torch (a change to your install, not made) or a YOLO ONNX → TensorRT detector path (not implemented).
* FP16 passes the accuracy checks but gives only a small end-to-end gain here, so FP32 stays the default.

### 5.5 Video output encoding

Output encoder candidates (`PS2_VIDEO_CODECS`, default `H264@MSMF:.mp4,mp4v:.mp4,MJPG:.avi`) must pass a **preflight** before use: they write 3 frames to a scratch file, reopen and decode them.

* `H264@MSMF`: H.264 through Windows Media Foundation, part of Windows. Browser-playable MP4.
* `mp4v`: MPEG-4 Part 2. Plays in VLC, not in browsers.
* `MJPG/.avi`: last resort.

**OpenH264 error, root cause:** `Failed to load OpenH264 library: openh264-2.5.0-win64.dll` is printed by OpenCV's *FFmpeg* backend when it is asked for H.264 (`avc1`). The opencv-python wheel delegates H.264 encoding to Cisco's DLL, which it doesn't ship. Reproduced: `cv2.VideoWriter(..., CAP_FFMPEG, 'avc1')` fails, while `CAP_MSMF` with `H264` produces a valid H.264 MP4. The default list never uses FFmpeg H.264, and no third-party DLL is downloaded. On non-Windows systems or Windows "N" editions without Media Foundation, the preflight fails and `mp4v` is used.

## 6. Tests

```powershell
python -m pytest tests -q                 # everything (~3 min; real models + real GPU where present)
python -m pytest tests -m "not hardware"  # skip real-backend tests
python -m pytest tests -m hardware -v -rs # only real-backend tests, with skip reasons
```

Last full run (2026-10-14, this machine, after the `source/` restructure): **124 passed, 3 skipped** (offline-evaluation tests removed together with `evaluation/`; layout/worker/readiness regressions in `tests/test_structure.py`). 9 of them are `hardware` tests (real TensorRT, ONNX-CPU, PyTorch-CPU). The 3 skips are `onnx-cuda`, `onnx-dml` and `pytorch-cuda`, whose runtimes are not installed here (onnxruntime has only the CPU EP; torch is a CPU build). They were not run, not mocked as passing. The real API end-to-end check (`scripts/e2e_api_test.py`) is listed in §7.1.

What is covered, and how:

| Area | File | Real or mocked |
|---|---|---|
| Backend selection: TensorRT available / package missing / engine corrupted / inference fails → fallback; ORT CUDA available; CUDA unavailable; CPU-only; PyTorch-only; ORT unavailable; no backend; wrong numerics / NaN rejected; runtime fallback mid-job; registry caching | `test_backend_selection.py` | **mocked** fakes + patched dependency probes: proves the logic, **not** that a GPU works |
| Every usable backend loads the same model, gives (N,512) finite unit-norm embeddings that match PyTorch (FP32 cos ≥ 0.99999, FP16 ≥ 0.999); same nearest neighbours; batching invariance; corrupted engine rejected; stale/tampered engine detected; auto-selection verified | `test_backend_hardware.py` (`hardware`) | **real** TensorRT on the RTX 2050, ONNX-CPU, PyTorch-CPU; unavailable runtimes are *skipped with the reason* |
| Identity preservation: same detections from 240 real frames tracked with PyTorch vs TensorRT vs ONNX → identical (frame, ID) rows | `test_backend_hardware.py` | real |
| Video: H.264 (Media Foundation) writer, fallbacks, preflight rejects broken encoder, no encoder; output validation; probe of missing/empty/text/still-image files; full video (30 and 450 frames); unreadable frame and decoder exception skipped; truncated file flagged; no people; writer failure | `test_video_io.py` | real OpenCV I/O, fake detector/ReID |
| Jobs/API: upload with only a video, poll, download, `/tracks`; legacy tuning fields ignored (whole video processed); OpenAPI exposes only `video`; invalid uploads (415/400) leave nothing behind; failed jobs remove partial outputs; actionable model-init errors; worker exceptions → failed (no path leaks); missing input; repeated jobs; 4 concurrent submissions; input cleanup; stale-file sweep | `test_video_io.py` | real FastAPI/worker, fake models |
| API with real detector + auto-selected backend: health/selection report, full 596-frame sample video, output decodes to 596 frames, track IDs consistent | `test_api.py` | real |
| Tracker logic, Kalman, association | `test_tracker.py`, `test_geometry_kalman.py` | synthetic |
| Layout, root worker CLI, `/ready`, `.env` loading, start from any directory | `test_structure.py` | real worker pipeline, fake models |

## 7. Measured results and limitations

### 7.1 Public CCTV-style test videos (through the real API)

Source: **MOTChallenge MOT17** training sequences, raw videos from `https://motchallenge.net/sequenceVideos/<SEQ>-raw.mp4`, ground truth from `https://motchallenge.net/data/MOT17Labels.zip`. The data is licensed **CC BY-NC-SA 3.0** according to motchallenge.net: attribution required, non-commercial, share-alike. Cite Milan et al., *MOT16: A Benchmark for Multi-Object Tracking*, arXiv:1603.00831, and Dendorfer et al., IJCV 2021. The files were downloaded to `D:\ps2_testdata\mot17` (outside the repo, not committed). The videos are 960×540 @ 30 FPS, and their decoded frame counts equal the GT `seqLength` (525 / 1050). GT boxes (1920×1080) were scaled ×0.5 for scoring, which leaves IoU unchanged.

Each video was uploaded to the running server via `scripts/e2e_api_test.py` with only the file. The job completed, the output was downloaded and decoded (frame count equal to the input), and `/jobs/{id}/tracks` was scored with TrackEval (offline evaluation tooling, no longer part of this repository).

| Sequence | Scene | Frames | Backend | IDs | HOTA | IDF1 | MOTA | IDSW | Frag | Recall | Precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-09 | static camera, pedestrian street | 525 | TensorRT FP32 | 24 | **44.6** | **56.8** | 58.3 | 27 | 76 | 63.1 % | 93.6 % |
| MOT17-04 | static elevated camera, dense crowd | 1050 | TensorRT FP32 | 36 | **32.8** | **34.3** | 20.6 | 6 | 88 | 22.7 % | 91.7 % |

Same detector and video (MOT17-09), tracker variants (same offline TrackEval scoring):

| Variant | HOTA | AssA | IDF1 | MOTA | IDSW | Frag |
|---|---|---|---|---|---|---|
| baseline (IoU/Kalman, SORT-like) | 40.5 | 37.8 | 51.4 | 51.3 | 54 | 97 |
| enhanced (this tracker) | 44.6 | 40.9 | 56.8 | 58.3 | 27 | 76 |

Reading these numbers honestly:

* On real data the enhanced tracker halves ID switches compared with the baseline.
* The API run and the direct pipeline run produce identical metrics, so the pipeline is deterministic.
* MOT17-04 is limited by **detection**: `yolo11n` at 640 px finds only 23 % of the small, distant pedestrians. Association quality there is reasonable (AssA 54.9, 6 IDSW), but HOTA/MOTA are low. A larger detector or a larger `PS2_DETECTOR_IMGSZ` is the lever, not ReID.
* These are two training sequences, not a benchmark submission.

### 7.2 Other measurements

- **Sample video** (`samples/people-detection.mp4`, 596 frames, fisheye hallway, no GT): 7 IDs. An earlier manual check found 7 distinct people with consistent IDs.
- **ReID domain gap:** on fisheye footage the PS-1 model separates identities weakly (ROC AUC ≈ 0.60), so long-gap recovery is deliberately strict (`reid_thresh` 0.25 + ratio test). Expect more new IDs rather than wrong merges.
- **Backend equivalence:** TensorRT FP32/FP16 and ONNX-CPU produce identical tracks to PyTorch on 240 real frames (test above).
- **Not measured:** ONNX Runtime CUDA/DirectML and PyTorch-CUDA. Those packages are not installed here; the code paths exist and are selection-tested with mocks only. Jetson was not measured either.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` → `degraded`, detector error | Put a YOLO `.pt` at `source/models/detector/yolo11n.pt` or set `PS2_DETECTOR_WEIGHTS` |
| `reid.error: No compatible ReID backend could be loaded: ...` | The message lists each candidate's reason. Usually the checkpoint is missing at `source/models/reid/best.pth`, or the ONNX model is missing (`python scripts\export_reid_onnx.py`) |
| Selected backend is `onnx-cpu` although there is an NVIDIA GPU | See the WARNING / `/health → reid.selection.attempts`. Typically TensorRT is not installed (§5.3) or `.gpu_runtime` was moved (set `PS2_GPU_RUNTIME`) |
| `TensorRT engine is stale (...)` in the log | Expected after a TensorRT/driver/GPU/ONNX change; it is rebuilt automatically (~17 s), or run `scripts\build_reid_tensorrt.py` |
| `Failed to deserialize the TensorRT engine` | Engine corrupted or from another machine; delete `source/models/tensorrt/*.engine*` (it is rebuilt) |
| `pip install tensorrt-cu12==10.x` fails on Python 3.14 | No cp314 bindings before TensorRT 11.1; use 11.4 |
| `OSError: [WinError 112] not enough space` while installing TensorRT | Install with `--target .gpu_runtime` and `TMP` on a drive with ≥ 8 GB free (§5.3) |
| `Failed to load OpenH264 library` | Only happens if `PS2_VIDEO_CODECS` includes FFmpeg `avc1`; use the default `H264@MSMF` |
| Output is `.avi` / `mp4v` | Media Foundation H.264 unavailable (non-Windows or "N" edition); install the Media Feature Pack, or play the file in VLC |
| 415 / 400 on upload | Unsupported extension, or not a decodable video (text, single image, header-less fragment). The message says which |
| Job `completed` with a `warnings` entry about truncation | The file ends before its header's frame count (interrupted copy); every decodable frame was processed |
| 507 on upload | Server disk full / not writable (`uploads/`, `uploads/.tmp`) |
| 429 | Too many pending jobs; wait for queued jobs to finish |
| `ConnectionResetError: [WinError 10054]` in `_call_connection_lost`, or `Accept failed on a socket [WinError 64]` followed by refused connections | Server started on the Windows Proactor loop; start it with `python source/main.py` or `--loop asyncio:SelectorEventLoop` (§3) |
| Too many new IDs after occlusion | Expected with the weak ReID on unseen domains; tune `TrackerConfig.reid_thresh` / `track_buffer` in `source/config/settings.py` (internal) and validate on GT |

## Layout

```
worker.py              standalone CLI worker (reuses source/logic; no duplicate pipeline)
requirements.txt  pytest.ini  .env.example  README.md
source/
  main.py              FastAPI app: start-up (model warm-load), routes, error handling, run()
  api/routes.py        /track /jobs/* /health /ready (input + video validation)
  config/settings.py   all paths (absolute, project-root based), model/tracker/job settings, .env loader
  logic/               shared business logic (API + worker)
    video_processor.py   decode -> detect -> ReID -> associate -> render -> write -> validate
    inference_manager.py automatic ReID backend selection, verification, runtime fallback
    job_manager.py       in-process job queue, status/progress persistence, cleanup
    tracking/            kalman_filter.py · association.py · track_manager.py · reid_extractor.py
  models/
    model_registry.py    backend factories + process-wide detector/ReID cache
    detector/            yolo11n.pt · inference.py (YOLO person detector)
    reid/                best.pth · inference.py (preprocessing, base backend, PyTorch backend)
    onnx/                reid_resnet18.onnx · inference.py (ONNX Runtime backend)
    tensorrt/            *.engine(.json) · reid_resnet18_fp16.onnx · inference.py · engine_builder.py
  utils/               device_utils.py (torch device, CUDA driver, .gpu_runtime) · video_utils.py
                       (writer + preflight + output validation) · logging_utils.py
scripts/     export_reid_onnx.py · build_reid_tensorrt.py · validate_backends.py · e2e_api_test.py
             (model maintenance: ONNX export, TensorRT engine build, backend check, API e2e test)
tests/       pytest suite (test_structure.py: layout / worker / readiness regressions)
samples/  uploads/  outputs/  .gpu_runtime/ (optional, git-ignored)
```

Dependency direction: `api` → `logic` → `models` → `utils` → `config`. `worker.py` and `source/main.py` both build on `logic/` and `models/model_registry.py`; the API runs jobs through `logic/job_manager.py`, the CLI worker runs one job directly through `logic/video_processor.py`.

