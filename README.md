# qwen3-asr-server

OpenAI-compatible transcription API server powered by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), fine-tuned on P25 public safety dispatch audio. Supports word-level timestamps via [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B).

Drop-in replacement for whisper-server — any client that talks to the OpenAI `/v1/audio/transcriptions` endpoint works out of the box.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (~4GB VRAM for both models at bfloat16)
- `ffmpeg` (required by librosa for non-wav audio formats)

## Quick Start

The `start.sh` script handles venv creation, dependency installation, and server startup:

```bash
git clone https://github.com/YOUR_USER/qwen3-asr-server.git
cd qwen3-asr-server
```

### 1. Download model weights

Download both models from Hugging Face into the repo directory:

```bash
# ASR model (fine-tuned on P25 audio)
git lfs install
git clone https://huggingface.co/YOUR_USER/qwen3-asr-p25-0.6B

# Forced aligner (for word-level timestamps)
git clone https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
```

You should end up with:
```
qwen3-asr-server/
├── qwen3-asr-p25-0.6B/       # ~1.5GB
├── Qwen3-ForcedAligner-0.6B/  # ~1.8GB
├── server.py
├── start.sh
└── ...
```

### 2. Start the server

```bash
./start.sh
```

This will:
1. Create a Python venv in `.venv/` (if it doesn't exist)
2. Install dependencies from `requirements.txt`
3. Start the server on port 8765

You can also pass arguments:
```bash
./start.sh /path/to/model 9000
```

### Manual setup

If you prefer to set things up yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

## Configuration

All settings are controlled via environment variables. Copy the example config to get started:

```bash
cp .env.example .env
# Edit .env as needed
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `qwen3-asr-p25-0.6B` | Path to ASR model directory |
| `ALIGNER_PATH` | `Qwen3-ForcedAligner-0.6B` | Path to forced aligner directory |
| `DEVICE` | `cuda:0` | Torch device (`cuda:0`, `cuda:1`, `cpu`) |
| `DTYPE` | `bfloat16` | Model precision (`bfloat16`, `float16`, `float32`) |
| `MAX_NEW_TOKENS` | `512` | Max generated tokens per transcription |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |
| `WORKERS` | `1` | Uvicorn workers (keep 1 unless multi-GPU) |
| `SPEECH_RMS_THRESHOLD` | `0.01` | RMS energy gate — audio below this is skipped as silence/encrypted |
| `REPETITION_THRESHOLD` | `4` | Reject if any n-gram repeats this many times (decoding loop detection) |
| `INFERENCE_TIMEOUT` | `30` | Per-request inference timeout in seconds |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `15` | Seconds to drain in-flight requests on shutdown |

## API

### `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint.

```bash
# Basic transcription
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr-p25

# With word-level timestamps
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr-p25 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"

# Plain text response
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=text
```

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | *(required)* | Audio file (wav, m4a, mp3, etc.) |
| `model` | string | `qwen3-asr-p25` | Model name (ignored, for API compat) |
| `language` | string | `English` | Language code (`en`, `zh`, `fr`, etc.) or full name |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `word_timestamps` | bool | `false` | Enable word-level timestamps |
| `timestamp_granularities[]` | list | — | Set to `word` to enable timestamps |

**Response formats:**

`json` (default):
```json
{"text": "All units respond to 5th and Main."}
```

`verbose_json`:
```json
{
  "task": "transcribe",
  "language": "English",
  "text": "All units respond to 5th and Main.",
  "processing_time": 0.832,
  "model": "qwen3-asr-p25-0.6B",
  "words": [
    {"word": "All", "start": 0.24, "end": 0.48},
    {"word": "units", "start": 0.52, "end": 0.88}
  ]
}
```

`text`:
```
All units respond to 5th and Main.
```

### `GET /v1/models`

Lists the loaded model. Compatible with OpenAI model listing.

### `GET /health`

Returns server status, model info, and current configuration.

## Speech Detection

The server includes an RMS energy gate to reject blank, silent, or encrypted P25 audio before it reaches the GPU. This prevents the model from hallucinating transcriptions on empty channels.

Threshold tuning (based on P25 radio):
- Encrypted/blank channels: RMS < 0.003
- Real speech: RMS > 0.03
- Default threshold: 0.01

Adjust `SPEECH_RMS_THRESHOLD` if you're working with different audio sources.

## Hallucination Detection

Known hallucination phrases (e.g., "thank you for watching", "hello how are you doing today...") are matched against the normalized transcription output and rejected with an empty string.

## Repetition Loop Detection

Detects when the model enters a decoding loop (e.g., "Engine 5 Engine 5 Engine 5 Engine 5"). Checks n-grams of size 1-4 and rejects if any pattern repeats `REPETITION_THRESHOLD` (default 4) or more times. Only checks texts with 8+ words to avoid false positives on short, legitimate dispatch audio. The threshold of 4 avoids false positives — dispatch operators legitimately repeat unit names 2-3 times.

## Inference Timeout

Each request has an `INFERENCE_TIMEOUT` (default 30s) safety net. Normal inference completes in <2s; the timeout only catches pathological cases where the model gets stuck in a decoding loop. On timeout, an empty transcription is returned and the event is logged.

## Logging

All events use Python's `logging` module at structured format:
```
2026-03-02 14:30:01,234 [12345] INFO OK file=call_123.wav words=15 time=0.832s
2026-03-02 14:30:02,567 [12345] WARNING REJECT rms_gate file=call_456.wav rms=0.0021 threshold=0.01 time=0.012s
```

Rejection events (`REJECT rms_gate`, `REJECT hallucination`, `REJECT repetition_loop`, `REJECT timeout`) are logged at WARNING level. Successful transcriptions (`OK`) at INFO level.

The `/health` endpoint includes per-worker counters tracking each outcome.

## Graceful Shutdown

The server uses uvicorn's built-in `timeout_graceful_shutdown` to drain in-flight requests before exiting. Default is 15 seconds. The systemd service sets `TimeoutStopSec=30` to give uvicorn time to finish draining before systemd sends SIGKILL.

## Running as a Service

An example systemd unit is included. Edit the paths and user in `systemd/qwen3-asr.service`, then:

```bash
sudo cp systemd/qwen3-asr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-asr
```

## Supported Languages

English, Chinese, Cantonese, Japanese, Korean, French, German, Spanish, Portuguese, Italian, Russian, Arabic, Indonesian — and more (see model config for the full list). Pass either ISO 639-1 codes (`en`, `zh`, `fr`) or full names (`English`, `Chinese`, `French`).
