# EXPAI Inference Platform

Real-time computer-vision inference platform for industrial use cases. Built on YOLOv8/v11 (Ultralytics) with ByteTrack tracking, FastAPI streaming over Server-Sent Events, and a single-page operator UI. Designed for GPU deployment on a fixed industrial PC or a cloud VM.

> Developed as part of the **EUREKA 21028 EXPAI** project — *Industrial Explainable AI for Process Quality and Operator Safety*.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Use-case modules](#use-case-modules)
3. [Repository layout](#repository-layout)
4. [Core runtime](#core-runtime)
5. [HTTP API](#http-api)
6. [Real-time streaming protocol (SSE)](#real-time-streaming-protocol-sse)
7. [Module contract (`BaseModule`)](#module-contract-basemodule)
8. [Configuration](#configuration)
9. [Installation](#installation)
10. [Running the platform](#running-the-platform)
11. [Docker deployment](#docker-deployment)
12. [GPU deployment notes](#gpu-deployment-notes)
13. [Output artifacts](#output-artifacts)
14. [Tracking & deduplication](#tracking--deduplication)
15. [Extending: adding a new module](#extending-adding-a-new-module)
16. [Project status](#project-status)

---

## Architecture overview

```
                       ┌──────────────────────────────────────────────┐
   Operator UI ◀──────▶│           FastAPI (api/main.py)              │
   (ui/index.html)     │                                              │
                       │  /modules    /upload     /inference          │
   Browser EventSource │  /results    /inspection                     │
   ◀──── SSE ──────────│                                              │
                       └─────────────────┬────────────────────────────┘
                                         │
                                         ▼
                       ┌──────────────────────────────────────────────┐
                       │       Inference orchestrator (background)    │
                       │                                              │
                       │  VideoProcessor ──▶ InferenceEngine ──▶      │
                       │   (OpenCV)          (Ultralytics YOLO        │
                       │                     + ByteTrack)             │
                       │                              │               │
                       │                              ▼               │
                       │                     module.process_frame()   │
                       │                     (analyzer + alerts)      │
                       │                              │               │
                       │           ┌──────────────────┼────────────┐  │
                       │           ▼                  ▼            ▼  │
                       │       SSE event         annotated      JSON  │
                       │       (queue)             video        report│
                       └──────────────────────────────────────────────┘
```

Single-process FastAPI app. Each inference is a `BackgroundTask` that owns:

* a `VideoProcessor` (OpenCV `VideoCapture`/`VideoWriter`),
* an `InferenceEngine` (Ultralytics YOLO wrapper with optional ByteTrack tracking),
* a use-case `module` implementing the `BaseModule` ABC,
* an `asyncio.Queue` that drains into the SSE response.

Inference runs in a thread executor (`loop.run_in_executor`) so the event loop stays responsive for the SSE consumer and the status endpoints.

---

## Use-case modules

| Module       | Classes detected                                                                                          | Purpose                                                                |
| ------------ | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `epp_gloves` | `con_guante`, `sin_guante`                                                                                | PPE compliance — detect operator hands without gloves.                 |
| `inactivity` | `estado_activo`, `estado_preparacion`, `estado_descarga`, `estado_inactivo`                               | Operator activity classification → inactivity incident timeline.       |
| `mixed`      | `mano_con_guante`, `mano_sin_guante`, `operario_activo`, `operario_transicion`                            | Combined PPE + activity. Critical risk = active operator + no gloves.  |
| `metrology`  | `bracket_ok`, `bracket_not_ok`                                                                            | Dimensional inspection (OK/NOK verdict) for stamped/bent parts.        |
| `kitting`    | `operario`, `bolsa_kit`, `kit_listo`, `hoja_etiquetas`                                                    | Count completed labelled kits via track-ID deduplication.              |
| `procesado`  | `pieza_ok`, `pieza_sin_proceso`                                                                           | Count processed pieces; dual alert (stall + backlog accumulation).     |

Each module is self-contained under `modules/<name>/`:

```
modules/<name>/
├── __init__.py
├── config.yaml      # Detection + analysis hyperparameters
├── analyzer.py      # Stateful per-frame logic (incidents, counters, transitions)
├── module.py        # Glue: implements BaseModule, wires analyzer + AlertSystem
└── alerts.py        # Optional — alert emission with cooldown
```

---

## Repository layout

```
expai-inference-platform/
├── api/
│   ├── main.py                # FastAPI app, CORS, router registration, /health
│   ├── schemas.py             # Pydantic request/response models
│   └── routes/
│       ├── inference.py       # /inference/start, /stream (SSE), /status, /stop
│       ├── inspection.py      # /inspection/photo — single-frame metrology
│       ├── modules.py         # /modules — list of available modules
│       ├── results.py         # /results/{job_id} — final JSON report + video
│       └── upload.py          # /upload/{model|video} — multipart upload
├── core/
│   ├── config.py              # Env + YAML loading, BASE_DIR, resolve_path
│   ├── inference_engine.py    # Detection class, InferenceEngine wrapper
│   ├── video_processor.py     # OpenCV VideoCapture/Writer + frame iterator
│   ├── visualizer.py          # BBox / status overlay drawing on frames
│   └── metrics.py             # Latency / FPS / confidence stats per session
├── modules/
│   ├── base.py                # BaseModule ABC (5 abstract methods)
│   ├── epp_gloves/  inactivity/  mixed/  metrology/  kitting/  procesado/
├── ui/
│   └── index.html             # Single-file SPA — modules grid, monitor, SSE consumer
├── tests/                     # Test directory (scaffolded)
├── main.py                    # CLI entry point (offline inference)
├── requirements.txt
├── Dockerfile                 # CUDA 12.2 runtime base
├── docker-compose.yml
├── setup_vm.sh                # GCP VM bootstrap script (CUDA, Python, service)
└── .env.example
```

---

## Core runtime

### `core/inference_engine.py`

Thin Ultralytics wrapper. The `Detection` dataclass is the single internal exchange type between the engine and modules:

```python
class Detection:
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    track_id:    int | None                 # populated only if enable_tracking=True
```

`InferenceEngine.predict(frame)` either calls `model.predict()` (stateless) or `model.track(persist=True, tracker="bytetrack.yaml")` (stateful, multi-object). Device resolution is `auto` (CUDA if available, else CPU).

### `core/video_processor.py`

Context-manager wrapper over OpenCV. Exposes:

* `.total_frames`, `.fps`, `.width`, `.height`
* `.frames()` — generator yielding `(frame_idx, timestamp_seconds, frame_bgr)`
* `.get_writer(path)` — `cv2.VideoWriter` matching source resolution/FPS

### `core/visualizer.py`

Draws YOLO bboxes with class label + confidence and a top-left status overlay (operator state, primary metric, alert banner). All annotations happen on a *copy* of the frame the writer consumes.

### `core/metrics.py`

`BaseMetrics` tracks per-session running averages: inference latency (ms), detections per frame, mean confidence, and uptime FPS.

---

## HTTP API

All routes are documented at `/docs` (Swagger UI) and `/redoc`. Summary:

| Method | Path                              | Purpose                                                       |
| ------ | --------------------------------- | ------------------------------------------------------------- |
| GET    | `/health`                         | Liveness check.                                               |
| GET    | `/modules`                        | List of registered modules (name, status, description).       |
| POST   | `/upload/{model\|video}`          | Multipart upload of `.pt` weights or video.                   |
| GET    | `/upload/{models\|videos}`        | List uploaded artifacts (size, path).                         |
| DELETE | `/upload/{type}/{filename}`       | Remove an uploaded artifact.                                  |
| POST   | `/inference/start`                | Start an inference job. Body: `module`, `video_path`, `model_path`, `config_overrides`. Returns `job_id`. |
| GET    | `/inference/{job_id}/status`      | Poll job status (`pending` / `running` / `completed` / `failed` / `stopping`). |
| GET    | `/inference/{job_id}/stream`      | **Server-Sent Events** stream of frame-by-frame updates.      |
| POST   | `/inference/{job_id}/stop`        | Request graceful stop (next frame boundary).                  |
| GET    | `/results/{job_id}`               | Final JSON report (analyzer summary + metrics + alerts).      |
| GET    | `/results/{job_id}/video`         | Download annotated MP4.                                       |
| POST   | `/inspection/photo`               | Single-frame metrology inspection (multipart image upload). Returns annotated image + verdict. |
| POST   | `/inspection/reload`              | Hot-reload the metrology model.                               |

---

## Real-time streaming protocol (SSE)

The browser opens an `EventSource` against `/inference/{job_id}/stream`. The backend pushes JSON payloads at every processed frame:

```json
{
  "frame": 1280,
  "timestamp": 51.2,
  "operator_state": "estado_activo",
  "alert_active": false,
  "alert_level": null,
  "compliance_pct": 96.4,
  "activity_rate": null,
  "inactive_streak": 0,
  "hand_state": null,
  "risk_level": null,
  "epp_compliance": null,
  "total_kits": null,
  "kits_per_min": null,
  "total_processed": null,
  "pending_count": null,
  "verdict": null,
  "events": [
    {"event_type": "EPP_VIOLATION", "timestamp": 51.2, "track_id": 7}
  ],
  "status": "running"
}
```

* All module-specific fields coexist in a single flat schema — only the relevant ones are populated per module. This keeps the UI consumer trivial.
* Keep-alive comments (`: keepalive\n\n`) are emitted every 30 s of silence.
* When the job finishes, a final message with `status: "completed"` and the full `report` payload is pushed before the stream closes.

The backing queue is bounded (`asyncio.Queue(maxsize=200)`). On overflow the producer drops the frame's SSE update (silently, via `put_nowait`) — the inference itself never blocks on a slow consumer.

---

## Module contract (`BaseModule`)

Every module implements `modules/base.py`:

```python
class BaseModule(ABC):
    @abstractmethod
    def load_config(self, config_path: str) -> None: ...

    @abstractmethod
    def process_frame(self, frame, detections, frame_index=0, timestamp=0.0) -> dict:
        # Returns: {"annotations": {...}, "events": [...], "metrics": {...}, "alert": {...}|None}
        ...

    @abstractmethod
    def get_state(self) -> dict: ...

    @abstractmethod
    def generate_report(self) -> dict: ...

    @abstractmethod
    def reset(self) -> None: ...
```

The orchestrator (`api/routes/inference.py::_run_inference`) calls `process_frame` for every frame and `generate_report` once on completion. The returned `annotations` dict directly drives the SSE payload and the on-frame status overlay.

---

## Configuration

Each module ships a `config.yaml` with the schema:

```yaml
model:
  path: models/your_model.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: auto            # auto | cpu | cuda:0

detection:
  enable_tracking: true
  tracker: bytetrack
  alert_frame_threshold: 15      # Module-specific
  critical_frame_threshold: 75   # Module-specific

analysis:
  rate_window_seconds: 60        # Module-specific

classes:
  0: con_guante
  1: sin_guante

output:
  annotated_video: true
  annotated_video_path: outputs/annotated_videos/
  report_path: outputs/reports/

alerts:
  enabled: true
  cooldown_seconds: 5.0
```

Per-job overrides can be passed in `POST /inference/start` via `config_overrides` — they are merged recursively into the base config.

Environment variables (see `.env.example`):

```
EXPAI_ENV=production
EXPAI_LOG_LEVEL=INFO
EXPAI_DATA_DIR=/mnt/data/expai-inference
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
```

---

## Installation

```bash
git clone https://github.com/<your-org>/expai-inference-platform.git
cd expai-inference-platform

python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env
# Edit .env as needed
```

Place your trained model weights in `models/` and test videos in `videos/` (both ignored by git).

---

## Running the platform

### REST API + UI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

* UI: <http://localhost:8000/>
* Swagger: <http://localhost:8000/docs>
* Health: <http://localhost:8000/health>

### CLI (offline batch inference)

```bash
python main.py \
  --module epp_gloves \
  --video videos/turno_manana.mp4 \
  --model models/epp_gloves_v1.pt \
  --output-dir outputs
```

Writes:

* `outputs/annotated_videos/<basename>_annotated.mp4`
* `outputs/reports/<basename>_report.json`

---

## Docker deployment

```bash
docker compose up --build -d
```

`Dockerfile` is based on `nvidia/cuda:12.2.0-runtime-ubuntu22.04`. For GPU passthrough you need the NVIDIA Container Toolkit installed on the host and either:

```yaml
# docker-compose.yml (already configured)
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
```

or `--gpus all` on `docker run`.

---

## GPU deployment notes

* **Model format.** Ultralytics auto-loads `.pt`. For maximum throughput export to TensorRT once:

  ```python
  from ultralytics import YOLO
  YOLO("models/your.pt").export(format="engine", half=True, device=0)
  ```

  Then point `model.path` to the resulting `.engine`. Expect ~2–4× FPS gain on a T4/L4 vs raw `.pt`.

* **Tracking cost.** ByteTrack adds a fixed per-frame cost (~1–2 ms). Disable `enable_tracking` for modules that don't need persistent IDs (e.g. `epp_gloves` if you only need per-frame compliance).

* **Frame skipping.** For long-running jobs on cheaper GPUs, `VideoProcessor.frames()` can be wrapped to process every Nth frame — extend if needed.

* **Single-worker FastAPI.** Use `--workers 1`. Multi-worker would duplicate every active model in VRAM.

* **`setup_vm.sh`** bootstraps a GCP `n1-standard-8` + T4/L4 VM end to end: NVIDIA drivers, CUDA, Python 3.11, repo clone, systemd service.

---

## Output artifacts

For every completed job:

```
outputs/
├── annotated_videos/<job_id>_annotated.mp4    # Source video + YOLO bboxes + status overlay
└── reports/<job_id>_report.json               # See structure below
```

Report structure (varies per module, common keys):

```json
{
  "module": "epp_gloves",
  "analysis": {
    "compliance_percentage": 96.4,
    "incident_count": 2,
    "incidents": [
      {"n":1,"start_timestamp":45,"end_timestamp":83,"duration_seconds":38,"track_id":7}
    ],
    "total_frames_analyzed": 14550,
    "avg_incident_duration_seconds": 38.0
  },
  "alerts":  { "total_emitted": 2, "by_level": {"WARNING": 2}, "history": [ ... ] },
  "metrics": { "avg_latency_ms": 23.1, "fps": 32.4, "mean_confidence": 0.86 }
}
```

---

## Tracking & deduplication

Counter-style modules (`kitting`, `procesado`) use ByteTrack persistent IDs to count each physical object exactly once:

```python
self._counted_track_ids: set[int] = set()

if det.class_name in self.processed_classes and det.track_id is not None:
    if det.track_id not in self._counted_track_ids:
        self._counted_track_ids.add(det.track_id)
        self.total_processed += 1
        new_events.append({"event_type": "PIEZA_PROCESADA", ...})
```

A `min_track_frames` threshold is applied to avoid counting fleeting false-positive tracks.

---

## Extending: adding a new module

1. **Scaffold.** Create `modules/<name>/` with `__init__.py`, `config.yaml`, `analyzer.py`, `module.py`.
2. **Implement `BaseModule`.** Subclass `modules.base.BaseModule`. The minimum is `process_frame()` returning `{"annotations": {...}, "events": [...], "metrics": {...}}`.
3. **Register.**
   * Add an entry in `api/routes/modules.py::AVAILABLE_MODULES`.
   * Add the import + dispatch in `api/routes/inference.py::_load_module()`.
   * Add it to `AVAILABLE_MODULES` (the set) in the same file.
4. **Wire the UI** (optional — schema is already permissive). If you add module-specific KPIs, extend `updateKpiLabels()` and `updateMonitor()` in `ui/index.html`.
5. **Train weights.** Drop the `.pt` into `models/` (or upload via `/upload/model`).

---

## Project status

Production-grade prototype validated on 5 active modules (`epp_gloves`, `inactivity`, `mixed`, `metrology`, `kitting`, `procesado`). Deployed on a GCP `n1-standard-8` + NVIDIA T4 VM.

**License:** TBD (project deliverable under EUREKA 21028 — contact the consortium for redistribution terms).
