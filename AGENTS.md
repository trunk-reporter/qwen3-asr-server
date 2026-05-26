# AGENTS.md

This file provides guidance to Codex and other coding agents when working with code in this repository. Legacy CLAUDE.md files may remain during migration; prefer AGENTS.md for active instructions.

## What This Is

A FastAPI server wrapping the Qwen3-ASR 0.6B model (fine-tuned on P25 dispatch audio) with an OpenAI-compatible transcription API. It also loads the Qwen3-ForcedAligner for word-level timestamps.

## Running the Server

```bash
# Quickstart (creates venv, installs deps, starts server)
./start.sh [MODEL_PATH] [PORT]

# Manual
source .venv/bin/activate
python server.py
```

All configuration is via environment variables (see `server.py` lines 36-48):
- `MODEL_PATH` (default: `qwen3-asr-p25-0.6B`) — local model directory
- `ALIGNER_PATH` (default: `Qwen3-ForcedAligner-0.6B`) — forced aligner directory
- `DEVICE` (default: `cuda:0`), `DTYPE` (default: `bfloat16`)
- `PORT` (default: `8765`), `HOST` (default: `0.0.0.0`), `WORKERS` (default: `1`)
- `SPEECH_RMS_THRESHOLD` (default: `0.01`) — RMS energy gate to skip blank/encrypted audio
- `MAX_NEW_TOKENS` (default: `512`)

## Architecture

Single-file server (`server.py`, ~230 lines). No tests, no build system.

**Key dependencies:** `qwen_asr` (provides `Qwen3ASRModel`), `transformers>=4.50`, `torch`, `librosa`, `fastapi`, `uvicorn`.

**Model loading:** On startup (`@app.on_event("startup")`), loads both the ASR model and the forced aligner onto GPU using `Qwen3ASRModel.from_pretrained()` with `forced_aligner` kwarg.

**Request flow for `POST /v1/audio/transcriptions`:**
1. Upload saved to temp file
2. `has_speech()` checks RMS energy via librosa — rejects silent/encrypted P25 audio (avoids GPU hallucinations)
3. If speech detected → `model.transcribe()` with optional `return_time_stamps=True`
4. Response formatted as `json` (OpenAI-compatible), `verbose_json` (with words/timing), or `text`

**Other endpoints:** `GET /v1/models`, `GET /health`

## Model Directories

- `qwen3-asr-p25-0.6B/` — Fine-tuned ASR model weights (~1.5GB safetensors)
- `Qwen3-ForcedAligner-0.6B/` — Forced aligner weights (~1.8GB safetensors)

Both are local checkpoints, not downloaded at runtime.

## P25-Specific Behavior

The speech detection gate (`SPEECH_RMS_THRESHOLD=0.01`) is tuned for P25 radio: blank/encrypted channels have RMS <0.003, real speech >0.03. This prevents hallucinated transcriptions on silent audio.
