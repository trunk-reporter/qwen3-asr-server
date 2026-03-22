# qwen3-asr-server

OpenAI-compatible transcription API server powered by [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), fine-tuned on P25 public safety dispatch audio. Supports word-level timestamps via [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B).

Drop-in replacement for whisper-server — any client that talks to the OpenAI `/v1/audio/transcriptions` endpoint works out of the box.

## Requirements

**Docker:** Just Docker (and `nvidia-container-toolkit` for GPU). No other dependencies needed.

**Native install:**
- Python 3.10+
- GPU: NVIDIA CUDA or Apple Silicon MPS (~4GB VRAM for both models at bfloat16). Falls back to CPU if neither is available.
- `ffmpeg` (for non-wav audio format conversion)

## Quick Start with Docker

The fastest way to get running. Models are downloaded automatically on first start.

### GPU (NVIDIA — word-level timestamps, fastest inference)

```bash
docker run --gpus all -p 8765:8765 \
  -v asr-model:/model -v asr-aligner:/aligner \
  ghcr.io/trunk-reporter/qwen3-asr-server:gpu
```

### CPU (no GPU required — uses C inference backend)

```bash
docker run -p 8765:8765 -v asr-model:/model \
  ghcr.io/trunk-reporter/qwen3-asr-server:cpu
```

Or use docker compose:

```bash
# GPU
docker compose -f docker-compose.gpu.yml up -d

# CPU
docker compose -f docker-compose.cpu.yml up -d
```

### Platform support

| Platform | Docker CPU | Docker GPU | Native (`./start.sh`) |
|----------|:----------:|:----------:|:---------------------:|
| Linux x86 + NVIDIA | yes | yes | yes |
| Linux ARM (Oracle, RPi) | yes | — | yes |
| macOS Apple Silicon | yes | — | yes (MPS) |
| macOS Intel | yes | — | yes (CPU) |

The GPU image requires NVIDIA + `nvidia-container-toolkit` on the host. macOS users who want GPU acceleration should run natively — MPS isn't available inside Docker containers.

### Docker configuration

Pass environment variables with `-e` to customize behavior:

```bash
docker run --gpus all -p 8765:8765 \
  -e SPEECH_RMS_THRESHOLD=0.02 \
  -e INFERENCE_TIMEOUT=60 \
  -v asr-model:/model -v asr-aligner:/aligner \
  ghcr.io/trunk-reporter/qwen3-asr-server:gpu
```

See the [Configuration](#configuration) section for all available variables.

### Building images locally

```bash
# CPU
docker build -t qwen3-asr-server:cpu .

# GPU
docker build -f Dockerfile.gpu -t qwen3-asr-server:gpu .
```

### Pre-downloaded models

If you already have the model weights locally, mount them directly instead of using Docker volumes:

```bash
# GPU
docker run --gpus all -p 8765:8765 \
  -v ./qwen3-asr-p25-0.6B:/model \
  -v ./Qwen3-ForcedAligner-0.6B:/aligner \
  ghcr.io/trunk-reporter/qwen3-asr-server:gpu

# CPU
docker run -p 8765:8765 \
  -v ./qwen3-asr-p25-0.6B:/model \
  ghcr.io/trunk-reporter/qwen3-asr-server:cpu
```

## Updating

### Docker

Pull the latest image and restart:

```bash
# GPU
docker pull ghcr.io/trunk-reporter/qwen3-asr-server:gpu
docker compose -f docker-compose.gpu.yml up -d

# CPU
docker pull ghcr.io/trunk-reporter/qwen3-asr-server:cpu
docker compose -f docker-compose.cpu.yml up -d
```

Or if running with `docker run`, just stop the container and re-run with the same command — the new image will be used.

Your model weights persist in the Docker volume, so they won't be re-downloaded.

### Native install

```bash
cd qwen3-asr-server
git pull
./start.sh
```

`start.sh` will automatically install any new dependencies before starting the server.

## Quick Start without Docker

### 1. Clone and download model weights

```bash
git clone https://github.com/trunk-reporter/qwen3-asr-server.git
cd qwen3-asr-server

# ASR model (fine-tuned on P25 audio)
git lfs install
git clone https://huggingface.co/AuggieActual/qwen3-asr-p25-0.6B

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
| `INFERENCE_BACKEND` | `python` | `python` (GPU, word timestamps) or `c` (CPU, no torch needed) |
| `C_BINARY_PATH` | `./qwen_asr` | Path to antirez/qwen-asr binary (C backend only) |
| `MODEL_PATH` | `qwen3-asr-p25-0.6B` | Path to ASR model directory |
| `ALIGNER_PATH` | `Qwen3-ForcedAligner-0.6B` | Path to forced aligner directory |
| `DEVICE` | `auto` | Torch device — `auto` picks CUDA > MPS > CPU; or pin with `cuda:0`, `mps`, `cpu` |
| `DTYPE` | `bfloat16` | Model precision (`bfloat16`, `float16`, `float32`) |
| `MAX_NEW_TOKENS` | `512` | Max generated tokens per transcription |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8765` | Server port |
| `WORKERS` | `1` | Uvicorn workers (keep 1 unless multi-GPU) |
| `SPEECH_RMS_THRESHOLD` | `0.01` | RMS energy gate — audio below this is skipped as silence/encrypted |
| `REPETITION_THRESHOLD` | `4` | Reject if any n-gram repeats this many times (decoding loop detection) |
| `INFERENCE_TIMEOUT` | `120` | Per-request inference timeout in seconds (CPU-safe default) |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `15` | Seconds to drain in-flight requests on shutdown |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | *(unset)* | Cap MPS memory allocations (0.0–1.0). Set to `0.7` on memory-constrained Macs. |

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

# With context prompt to help with spelling
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr-p25 \
  -F prompt="Engine 12, Ladder 7, Elmhurst Avenue"

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
| `prompt` | string | — | Optional context to guide transcription (see [Prompt / Context](#prompt--context) below) |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `word_timestamps` | bool | `false` | Enable word-level timestamps (Python backend only) |
| `timestamp_granularities[]` | list | — | Set to `word` to enable timestamps (Python backend only) |

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

Returns server status, model info, current configuration, and per-worker request counters.

## Prompt / Context

The `prompt` parameter lets you pass context to the model to influence transcription. It's injected into the model's system prompt to nudge token probabilities toward specific terms. Works with both backends — maps to `context` in the Python backend and `--prompt` in the C backend.

**Use cases:**

- **Spelling and proper nouns** — nudge the model toward domain-specific names it wouldn't otherwise get right
- **Formatting style** — hint at preferred output conventions
- **Domain vocabulary** — provide jargon or abbreviations common to your audio source

**Examples:**

```bash
# Help with local street names and unit designations
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@dispatch.wav \
  -F model=qwen3-asr-p25 \
  -F prompt="Rensselaer County, Engine 45, Pawling Avenue, Taconic Parkway"

# Abbreviation and formatting hints
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@dispatch.wav \
  -F model=qwen3-asr-p25 \
  -F prompt="Use standard abbreviations: EMS, CPR, MVA, DOA"

# With Python OpenAI client
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F file=@dispatch.wav \
  -F model=qwen3-asr-p25 \
  -F prompt="St. Clair Shores PD, Lake Shore Drive, Jefferson Avenue"
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="not-needed")

transcript = client.audio.transcriptions.create(
    model="qwen3-asr-p25",
    file=open("dispatch.wav", "rb"),
    prompt="Engine 12, Ladder 7, Battalion 3, Elmhurst Avenue",
)
print(transcript.text)
```

**Caveats:** Prompt biasing is "very soft" — the model may or may not follow your instructions. Spelling hints tend to work best. Strong or lengthy prompts can bias the model heavily, causing it to parrot the prompt instead of reflecting the actual audio. Long prompts also increase sequence length and memory/compute per request. Start with short, factual context (proper nouns, abbreviations) rather than full sentences. If results get worse, try shortening the prompt or removing it entirely.

## Inference Backends

The server supports two inference backends:

**Python** (`INFERENCE_BACKEND=python`, default) — Uses PyTorch + transformers with GPU acceleration (CUDA or MPS). Supports word-level timestamps via the ForcedAligner. This is the full-featured backend for production use with a GPU.

**C** (`INFERENCE_BACKEND=c`) — Uses the [antirez/qwen-asr](https://github.com/antirez/qwen-asr) C binary. No GPU, no PyTorch, no CUDA required. The model weights are memory-mapped and inference runs on CPU via OpenBLAS. The Docker CPU image is ~800MB vs multi-GB for the GPU image. Trade-off: no word-level timestamps.

| | Python backend | C backend |
|---|---|---|
| GPU acceleration | CUDA, MPS | — |
| Word-level timestamps | yes | no |
| Docker image size | ~8GB | ~800MB |
| Dependencies | torch, transformers, qwen_asr | just the C binary |
| Typical inference time | <2s (GPU) | ~2-5s (CPU) |

## Device Auto-Detection

*Applies to the Python backend only. The C backend always runs on CPU.*

The server automatically selects the best available device at startup:

1. **CUDA** (NVIDIA GPU) — preferred on Linux/Windows
2. **MPS** (Apple Silicon) — preferred on macOS 12.3+
3. **CPU** — universal fallback

Set `DEVICE=auto` (the default) for automatic selection, or pin a specific device with `DEVICE=cuda:0`, `DEVICE=mps`, or `DEVICE=cpu`. If the requested device isn't available, the server falls back to CPU with a warning.

On Apple Silicon, the server automatically flushes the MPS memory cache after each request to prevent memory accumulation. If you hit memory limits, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` to cap MPS allocations. The server also probes bfloat16 support on MPS at startup and falls back to float16 if needed.

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

Each request has an `INFERENCE_TIMEOUT` (default 120s) safety net. GPU inference completes in <2s; CPU inference can take 30-60s for longer audio files. The 120s default is safe for CPU deployments while still catching truly pathological decoding loops. On timeout, an empty transcription is returned and the event is logged.

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
