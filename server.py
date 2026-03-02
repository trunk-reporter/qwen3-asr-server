"""
Qwen3-ASR P25 transcription server.

OpenAI-compatible API for Qwen3-ASR fine-tuned on P25 dispatch audio.
Mirrors the whisper-server API pattern for drop-in compatibility.
Supports word-level timestamps via Qwen3-ForcedAligner.

Usage:
    python server.py

    # Or with env vars:
    MODEL_PATH=/path/to/model PORT=8765 python server.py

Endpoints:
    POST /v1/audio/transcriptions  — OpenAI-compatible transcription
    GET  /v1/models                — List loaded model
    GET  /health                   — Health check
"""

import os
import re
import tempfile
import time
from typing import Optional

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from qwen_asr import Qwen3ASRModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "qwen3-asr-p25-0.6B")
ALIGNER_PATH = os.environ.get("ALIGNER_PATH", "Qwen3-ForcedAligner-0.6B")
_DEVICE_RAW = os.environ.get("DEVICE", "auto")
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
WORKERS = int(os.environ.get("WORKERS", "1"))

# Speech detection — reject blank/encrypted audio before wasting GPU
# RMS energy threshold: audio below this is silence/static/encrypted
# Empirically: hallucinations <0.003 RMS, real speech >0.03 RMS
SPEECH_RMS_THRESHOLD = float(os.environ.get("SPEECH_RMS_THRESHOLD", "0.01"))

# Hallucination detection — known phrases that models produce on noise/static
# that passes the RMS gate but contains no intelligible speech.
# Normalized to lowercase alphanumeric for fuzzy matching.
_HALLUCINATION_PHRASES = {
    "hello how are you doing today the weather is beautiful and the world seems full of possibilities",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "thank you for listening",
    "thanks for listening",
    "please like and subscribe",
    "the end",
    "you",
    "bye",
}

def _normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for hallucination matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def is_hallucination(text: str) -> bool:
    """Check if transcription is a known hallucination phrase.

    Uses normalized substring matching — if the entire transcription output
    matches a known hallucination phrase, it's rejected. Short outputs (<3 words)
    that aren't plausible dispatch content are also caught.
    """
    norm = _normalize_for_match(text)
    if not norm:
        return False
    return norm in _HALLUCINATION_PHRASES

# MPS memory management — cap MPS allocations to this fraction of system RAM.
# Lowering this causes MPS to raise an error earlier rather than crashing hard.
# Set via PYTORCH_MPS_HIGH_WATERMARK_RATIO in .env or environment.
# Default PyTorch value is 1.0 (no cap); 0.7 is a safe starting point.
_MPS_WATERMARK = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
if _MPS_WATERMARK is not None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = _MPS_WATERMARK

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Map ISO 639-1 / common short codes to Qwen3-ASR language names
LANG_MAP = {
    "en": "English", "english": "English",
    "zh": "Chinese", "chinese": "Chinese",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "ar": "Arabic", "arabic": "Arabic",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "es": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "id": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "italian": "Italian",
    "ko": "Korean", "korean": "Korean",
    "ru": "Russian", "russian": "Russian",
    "ja": "Japanese", "japanese": "Japanese",
}

# ---------------------------------------------------------------------------
# Device + dtype resolution
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    """
    Resolve the best available device.

    When DEVICE=auto (the default), priority is:
      1. CUDA  (NVIDIA GPU — Linux / Windows)
      2. MPS   (Apple Silicon — macOS 12.3+)
      3. CPU   (universal fallback)

    You can still pin a device explicitly:
      DEVICE=cuda:0   DEVICE=mps   DEVICE=cpu
    """
    req = requested.lower().strip()

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested but not available — falling back to CPU")
        return "cpu"

    if req == "mps" and not torch.backends.mps.is_available():
        print(f"WARNING: MPS requested but not available — falling back to CPU")
        return "cpu"

    return req


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    """
    Resolve dtype, with an automatic bfloat16 -> float16 fallback on MPS.

    bfloat16 on MPS requires macOS 14+ and PyTorch 2.2+.
    We probe it at startup and downgrade silently if it isn't supported.
    """
    dt = DTYPE_MAP.get(dtype_str.lower(), torch.bfloat16)

    if device == "mps" and dt == torch.bfloat16:
        try:
            torch.zeros(1, dtype=torch.bfloat16, device="mps")
            print("MPS bfloat16 probe: supported v")
        except (RuntimeError, TypeError):
            print("MPS bfloat16 not supported on this system — using float16 instead")
            dt = torch.float16

    return dt


def device_map_arg(device: str):
    """
    Convert a device string to the form expected by Qwen3ASRModel / transformers.

    transformers' device_map on MPS must be {"": "mps"} rather than the bare
    string "mps", because the accelerate dispatcher doesn't enumerate MPS as a
    named device the way it does for CUDA.  Using {"": "mps"} puts every layer
    on the single MPS device without going through accelerate's CUDA-specific
    multi-device path.

    For CPU we also use the dict form so the code path is consistent.
    For CUDA strings (e.g. "cuda:0") the bare string works fine.
    """
    if device in ("mps", "cpu"):
        return {"": device}
    return device  # e.g. "cuda:0" — leave as-is


def flush_mps_cache():
    """Release MPS memory after each inference call.

    Unlike CUDA, MPS does not automatically free cached allocations between
    calls, causing memory to accumulate across requests until the system runs
    out and crashes with a Metal buffer allocation error.
    """
    if DEVICE == "mps":
        torch.mps.empty_cache()


DEVICE = resolve_device(_DEVICE_RAW)
_DTYPE_OBJ = resolve_dtype(DTYPE, DEVICE)
_DEVICE_MAP = device_map_arg(DEVICE)

print(f"Device: {DEVICE}  |  dtype: {_DTYPE_OBJ}  |  device_map: {_DEVICE_MAP}")

# ---------------------------------------------------------------------------
# App + model
# ---------------------------------------------------------------------------
app = FastAPI(title="qwen3-asr-p25-server", version="1.3.0")
model: Optional[Qwen3ASRModel] = None


def has_speech(audio_path: str) -> bool:
    """Check if audio contains actual speech via RMS energy.

    Rejects blank/encrypted/silent audio that would cause hallucinations.
    Empirically: blank P25 audio <0.003 RMS, real speech >0.03 RMS.
    Uses librosa to support all audio formats (wav, m4a, etc.).
    """
    audio, _ = librosa.load(audio_path, sr=None, mono=True)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms >= SPEECH_RMS_THRESHOLD


@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model: {MODEL_PATH} (device={DEVICE}, dtype={DTYPE})")
    print(f"Loading aligner: {ALIGNER_PATH}")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        forced_aligner=ALIGNER_PATH,
        forced_aligner_kwargs=dict(dtype=_DTYPE_OBJ, device_map=_DEVICE_MAP),
        dtype=_DTYPE_OBJ,
        device_map=_DEVICE_MAP,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("Model + aligner loaded.")
    print(f"Speech detection: RMS threshold={SPEECH_RMS_THRESHOLD}")


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("qwen3-asr-p25", alias="model"),
    language: Optional[str] = Form("English"),
    response_format: str = Form("json"),
    word_timestamps: Optional[bool] = Form(None),
    timestamp_granularities: Optional[list[str]] = Form(None, alias="timestamp_granularities[]"),
):
    t0 = time.time()

    # Determine if timestamps requested
    want_timestamps = word_timestamps or bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    # Normalize language code
    lang = (language or "English").strip()
    lang = LANG_MAP.get(lang.lower(), lang)

    # Write upload to temp file
    data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Speech detection gate — skip GPU inference for blank/encrypted audio
        if not has_speech(tmp_path):
            processing_time = round(time.time() - t0, 3)
            full_text = ""
            words = []
        else:
            results = model.transcribe(
                audio=tmp_path,
                language=lang,
                return_time_stamps=want_timestamps,
            )

            r = results[0] if results else None
            full_text = r.text.strip() if r else ""
            processing_time = round(time.time() - t0, 3)

            # Hallucination filter — reject known bogus phrases
            if is_hallucination(full_text):
                full_text = ""
                words = []
            else:
                # Build word list from timestamps
                words = []
                if want_timestamps and r and r.time_stamps:
                    for item in r.time_stamps:
                        words.append({
                            "word": item.text,
                            "start": round(item.start_time, 3),
                            "end": round(item.end_time, 3),
                        })

    finally:
        os.unlink(tmp_path)
        flush_mps_cache()  # Release MPS memory after every request

    # --- Format response ---
    if response_format == "text":
        return PlainTextResponse(full_text)

    if response_format == "verbose_json":
        resp = {
            "task": "transcribe",
            "language": lang,
            "text": full_text,
            "processing_time": processing_time,
            "model": MODEL_PATH,
        }
        if want_timestamps:
            resp["words"] = words
        return JSONResponse(resp)

    # Default: json (OpenAI-compatible)
    return JSONResponse({"text": full_text})


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_PATH,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "aligner": ALIGNER_PATH,
        "device": DEVICE,
        "dtype": DTYPE,
        "workers": WORKERS,
        "pid": os.getpid(),
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
        "hallucination_phrases": len(_HALLUCINATION_PHRASES),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if WORKERS > 1:
        uvicorn.run("server:app", host=HOST, port=PORT, workers=WORKERS)
    else:
        uvicorn.run(app, host=HOST, port=PORT)
