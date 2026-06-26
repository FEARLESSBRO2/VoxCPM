# VoxCPM Android App — Phase 1 MVP Design

> Date: 2026-06-26
> Status: approved design, pre-implementation
> Scope: Phase 1 MVP only. Public hardening (accounts, quotas, autoscale, object storage, consent gate) is **Phase 2**, explicitly out of scope here.

## 0. Context

VoxCPM2 is a GPU-only multilingual TTS engine (2B params, CUDA + large VRAM required). A phone cannot run the model. The Android app is therefore a **thin client to a GPU backend**. See `brain.md` for the as-built system map.

This spec covers a single-user MVP: a FastAPI backend wrapping the existing `VoxCPM` class (no model code changes) and a native Kotlin/Compose Android client, authenticated by one shared static token.

## 1. Architecture

```
Android app (Kotlin + Jetpack Compose)
   │  HTTPS, Authorization: Bearer <token>
   ▼
FastAPI backend (uvicorn) on a RunPod / VPS GPU box
   │  loads VoxCPM once at startup (warm)
   ▼
VoxCPM (src/voxcpm/core.py) → CUDA inference → WAV
```

- The model is loaded **once at startup** and stays resident (warm). No per-request load.
- Single GPU → all generation is **serialized**: one worker thread holds a GPU lock and drains a job queue. Concurrent submits are accepted but processed one at a time.
- No model/library code changes. The backend imports and calls `VoxCPM` exactly as the Gradio app does.

## 2. Authentication

- Single static Bearer token, read from env var `VOXCPM_TOKEN` at startup.
- Every endpoint requires header `Authorization: Bearer <token>`. Mismatch → `401`.
- One shared identity for Phase 1. Real per-user accounts are Phase 2.

## 3. API contract

All endpoints require the Bearer header.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/voices` | save a reference voice (multipart: `wav` file + `name`) → returns `voice_id` |
| `GET` | `/voices` | list saved voices |
| `DELETE` | `/voices/{voice_id}` | delete a saved voice |
| `POST` | `/generate` | submit a generation job → returns `job_id` |
| `GET` | `/jobs/{job_id}` | job status + progress |
| `GET` | `/jobs/{job_id}/audio` | download the result WAV (once `done`) |

### 3.1 `POST /generate` request body

```json
{
  "mode": "tts | design | clone | ultimate",
  "text": "target text to synthesize",
  "voice_id": "optional — reference voice for clone/ultimate",
  "control": "optional — voice-design instruction (mode=design)",
  "prompt_text": "optional — transcript of reference (mode=ultimate)",
  "cfg": 2.0,
  "dit_steps": 10,
  "denoise": false,
  "normalize": false
}
```

Field-to-mode mapping:

| mode | uses |
|---|---|
| `tts` | `text` only (default voice) |
| `design` | `text` + `control` |
| `clone` | `text` + `voice_id` (reference wav) |
| `ultimate` | `text` + `voice_id` + `prompt_text` |

These map onto `VoxCPM._generate(...)` args: `voice_id` → `reference_wav_path`, `control` → control instruction, `prompt_text` → `prompt_text`, `cfg` → `cfg_value`, `dit_steps` → `inference_timesteps`, plus `denoise` / `normalize` passthrough.

### 3.2 `GET /jobs/{job_id}` response

```json
{
  "job_id": "...",
  "status": "queued | running | done | error",
  "progress": 0.0,
  "error": null
}
```

`progress` is a float `0.0–1.0` derived from chunks completed / total chunks.

## 4. Async job model

Long narration takes minutes — too long for a single blocking HTTP request. The flow is **submit → poll → download**:

1. Client `POST /generate` → backend creates a job, returns `job_id` immediately.
2. Backend worker thread picks the job, holds the GPU lock, runs generation.
3. Client polls `GET /jobs/{job_id}` until `status == "done"`.
4. Client `GET /jobs/{job_id}/audio` to download the WAV.

Backend internals:
- In-memory job registry: `{job_id: {status, progress, wav_path, error}}`.
- A single worker thread + a `queue.Queue`; the worker holds the GPU lock so only one generation runs at a time.
- **Server-side chunking**: port `split_text_into_chunks(text, max_chars=150)` from `app.py`. The client sends the full script; the backend chunks it, generates per chunk, stitches with 0.3s silence, and writes one WAV. `progress` updates per chunk.

Job registry is in-memory only (Phase 1). A backend restart loses in-flight jobs — acceptable for single-user MVP.

## 5. Storage

Phase 1 uses local disk on the GPU box:

- `voices/<voice_id>.wav` + a small JSON sidecar (or a single `voices/index.json`) holding `{voice_id, name}`.
- `outputs/<job_id>.wav` for results.

No object storage in Phase 1. S3/GCS migration is Phase 2.

## 6. Android client

Native Kotlin + Jetpack Compose.

Screens / components:
- **Mode tabs**: Text-to-Speech, Voice Design, Voice Cloning, Ultimate Cloning (parity with the Gradio app).
- **Voice picker**: list / save / delete via `/voices`; record a new reference with `MediaRecorder`.
- **Script input** + **advanced controls**: CFG, DiT steps, denoise toggle, normalize toggle.
- **Generate → poll → result**: progress bar driven by `GET /jobs/{id}`; on `done`, play with ExoPlayer and save/share the WAV via `MediaStore`.

Networking: Retrofit/OkHttp (or Ktor). Backend base URL + token configured in app settings.

## 7. Out of scope (Phase 2)

Accounts/signup, per-user isolation, GPU autoscale/serverless, quotas/rate-limit/billing, object storage, voice-cloning consent gate + abuse reporting. Noted here so Phase 1 is not over-built.

## 8. Success criteria

- All 4 modes work end-to-end from the Android app.
- A long script (>150 chars) is chunked server-side and returned as a single stitched WAV.
- A saved voice persists and can be reused across requests.
- A request without the correct Bearer token is rejected with `401`.
- A multi-minute generation completes via poll without an HTTP timeout.
