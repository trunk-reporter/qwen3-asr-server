# CPU inference image using antirez/qwen-asr C binary.
# No GPU required. ~800MB image.
#
# Build:
#   docker build -t qwen3-asr-server:cpu .
#
# Run (model auto-downloads on first start):
#   docker run -p 8765:8765 -v asr-model:/model qwen3-asr-server:cpu

# --- Stage 1: Build the C binary from antirez/qwen-asr ---
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/antirez/qwen-asr.git . && make blas

# --- Stage 2: Runtime ---
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser \
    && mkdir -p /model \
    && chown appuser:appuser /model

WORKDIR /app

# Install Python deps (C-backend — no torch needed)
RUN pip install --no-cache-dir \
    fastapi \
    'uvicorn[standard]' \
    python-multipart \
    soundfile \
    numpy \
    'huggingface_hub>=0.34'

# Copy C binary from builder
COPY --from=builder /build/qwen_asr /app/qwen_asr

# Copy server + entrypoint
COPY server.py entrypoint.sh ./

ENV INFERENCE_BACKEND=c \
    C_BINARY_PATH=/app/qwen_asr \
    MODEL_PATH=/model \
    ASR_MODEL_REPO=trunk-reporter/qwen3-asr-p25-0.6B \
    HOST=0.0.0.0 \
    PORT=8765 \
    WORKERS=1 \
    SPEECH_RMS_THRESHOLD=0.01

EXPOSE 8765

USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
