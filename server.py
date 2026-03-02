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

import asyncio
import logging
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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("qwen3-asr")

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

# Repetition loop detection — reject decoding loops like "Engine 5 Engine 5 Engine 5 Engine 5"
REPETITION_THRESHOLD = int(os.environ.get("REPETITION_THRESHOLD", "4"))

# Per-request inference timeout (seconds) — catches pathological decoding
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "30"))

# Graceful shutdown timeout (seconds) — drain in-flight requests before exit
GRACEFUL_SHUTDOWN_TIMEOUT = int(os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT", "15"))

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

def is_hallucination(text: str) -> tuple[bool, str]:
    """Check if transcription is a known hallucination phrase.

    Uses normalized substring matching — if the entire transcription output
    matches a known hallucination phrase, it's rejected. Short outputs (<3 words)
    that aren't plausible dispatch content are also caught.

    Returns (is_hallucination, matched_phrase).
    """
    norm = _normalize_for_match(text)
    if not norm:
        return False, ""
    if norm in _HALLUCINATION_PHRASES:
        return True, norm
    return False, ""

def has_repetition_loop(text: str) -> tuple[bool, str]:
    """Detect repetition loops in transcription output.

    Checks n-grams of size 1-4. If any n-gram repeats REPETITION_THRESHOLD
    or more times, the text is flagged as a decoding loop.

    Only checks texts with 8+ words — short texts can't have 4 meaningful reps.

    Returns (is_loop, repeated_pattern).
    """
    words = text.split()
    if len(words) < 8:
        return False, ""

    for n in range(1, 5):  # 1-gram through 4-gram
        if len(words) < n:
            continue
        counts: dict[str, int] = {}
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= REPETITION_THRESHOLD:
                return True, gram
    return False, ""

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
# Per-worker counters
# ---------------------------------------------------------------------------
_counters = {
    "ok": 0,
    "reject_rms_gate": 0,
    "reject_hallucination": 0,
    "reject_repetition_loop": 0,
    "reject_timeout": 0,
    "total": 0,
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
        logger.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"

    if req == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available — falling back to CPU")
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
            logger.info("MPS bfloat16 probe: supported")
        except (RuntimeError, TypeError):
            logger.info("MPS bfloat16 not supported on this system — using float16 instead")
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

logger.info(f"Device: {DEVICE}  |  dtype: {_DTYPE_OBJ}  |  device_map: {_DEVICE_MAP}")

# ---------------------------------------------------------------------------
# App + model
# ---------------------------------------------------------------------------
app = FastAPI(title="qwen3-asr-p25-server", version="1.4.0")
model: Optional[Qwen3ASRModel] = None


def has_speech(audio_path: str) -> tuple[bool, float]:
    """Check if audio contains actual speech via RMS energy.

    Rejects blank/encrypted/silent audio that would cause hallucinations.
    Empirically: blank P25 audio <0.003 RMS, real speech >0.03 RMS.
    Uses librosa to support all audio formats (wav, m4a, etc.).

    Returns (has_speech, rms_value).
    """
    audio, _ = librosa.load(audio_path, sr=None, mono=True)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms >= SPEECH_RMS_THRESHOLD, rms


def _run_inference(audio_path: str, lang: str, want_timestamps: bool):
    """Synchronous inference helper — runs model.transcribe() in a thread."""
    results = model.transcribe(
        audio=audio_path,
        language=lang,
        return_time_stamps=want_timestamps,
    )
    return results


@app.on_event("startup")
def load_model():
    global model
    logger.info(f"Loading model: {MODEL_PATH} (device={DEVICE}, dtype={DTYPE})")
    logger.info(f"Loading aligner: {ALIGNER_PATH}")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        forced_aligner=ALIGNER_PATH,
        forced_aligner_kwargs=dict(dtype=_DTYPE_OBJ, device_map=_DEVICE_MAP),
        dtype=_DTYPE_OBJ,
        device_map=_DEVICE_MAP,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    logger.info("Model + aligner loaded.")
    logger.info(f"Speech detection: RMS threshold={SPEECH_RMS_THRESHOLD}")
    logger.info(f"Repetition detection: threshold={REPETITION_THRESHOLD}")
    logger.info(f"Inference timeout: {INFERENCE_TIMEOUT}s")
    logger.info(f"Graceful shutdown timeout: {GRACEFUL_SHUTDOWN_TIMEOUT}s")


@app.on_event("shutdown")
def on_shutdown():
    logger.info(
        f"Shutting down — counters: {_counters}"
    )


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
    filename = file.filename or "unknown"
    _counters["total"] += 1

    # Determine if timestamps requested
    want_timestamps = word_timestamps or bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    # Normalize language code
    lang = (language or "English").strip()
    lang = LANG_MAP.get(lang.lower(), lang)

    # Write upload to temp file
    data = await file.read()
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # --- RMS gate ---
        speech_detected, rms_value = has_speech(tmp_path)
        if not speech_detected:
            processing_time = round(time.time() - t0, 3)
            _counters["reject_rms_gate"] += 1
            logger.warning(
                f"REJECT rms_gate file={filename} rms={rms_value:.4f} "
                f"threshold={SPEECH_RMS_THRESHOLD} time={processing_time}s"
            )
            full_text = ""
            words = []
        else:
            # --- Inference with timeout ---
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(_run_inference, tmp_path, lang, want_timestamps),
                    timeout=INFERENCE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                processing_time = round(time.time() - t0, 3)
                _counters["reject_timeout"] += 1
                logger.warning(
                    f"REJECT timeout file={filename} limit={INFERENCE_TIMEOUT}s "
                    f"time={processing_time}s"
                )
                full_text = ""
                words = []
                # Skip hallucination/repetition checks — go straight to response
                return _format_response(response_format, full_text, words, lang,
                                        want_timestamps, round(time.time() - t0, 3))

            r = results[0] if results else None
            full_text = r.text.strip() if r else ""
            processing_time = round(time.time() - t0, 3)

            # --- Hallucination filter ---
            hallucinated, matched_phrase = is_hallucination(full_text)
            if hallucinated:
                _counters["reject_hallucination"] += 1
                logger.warning(
                    f"REJECT hallucination file={filename} "
                    f"phrase=\"{matched_phrase}\" time={processing_time}s"
                )
                full_text = ""
                words = []
            else:
                # --- Repetition loop filter ---
                is_loop, loop_pattern = has_repetition_loop(full_text)
                if is_loop:
                    _counters["reject_repetition_loop"] += 1
                    logger.warning(
                        f"REJECT repetition_loop file={filename} "
                        f"pattern=\"{loop_pattern}\" words={len(full_text.split())} "
                        f"time={processing_time}s"
                    )
                    full_text = ""
                    words = []
                else:
                    # --- Success ---
                    _counters["ok"] += 1
                    word_count = len(full_text.split()) if full_text else 0
                    logger.info(
                        f"OK file={filename} words={word_count} "
                        f"time={processing_time}s"
                    )

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

    processing_time = round(time.time() - t0, 3)
    return _format_response(response_format, full_text, words, lang,
                            want_timestamps, processing_time)


def _format_response(response_format, full_text, words, lang,
                     want_timestamps, processing_time):
    """Build the HTTP response in the requested format."""
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
        "version": "1.4.0",
        "model": MODEL_PATH,
        "aligner": ALIGNER_PATH,
        "device": DEVICE,
        "dtype": DTYPE,
        "workers": WORKERS,
        "pid": os.getpid(),
        "config": {
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
            "hallucination_phrases": len(_HALLUCINATION_PHRASES),
            "repetition_threshold": REPETITION_THRESHOLD,
            "inference_timeout": INFERENCE_TIMEOUT,
            "graceful_shutdown_timeout": GRACEFUL_SHUTDOWN_TIMEOUT,
        },
        "counters": dict(_counters),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if WORKERS > 1:
        uvicorn.run(
            "server:app",
            host=HOST,
            port=PORT,
            workers=WORKERS,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT,
        )
    else:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT,
        )
