# Multi-stage build: compile qwen_asr C binary, then slim runtime image.
#
# Build (from repo root):
#   docker build -t qwen3-asr-server .
#
# Run (mount your model weights):
#   docker run -p 8765:8765 -v /path/to/qwen3-asr-p25-0.6B:/model qwen3-asr-server
#
# Or download model first:
#   huggingface-cli download AuggieActual/qwen3-asr-p25-0.6B --local-dir ./model
#   docker run -p 8765:8765 -v ./model:/model qwen3-asr-server

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
    ffmpeg \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (C-backend only — no torch needed)
RUN pip install --no-cache-dir \
    fastapi \
    'uvicorn[standard]' \
    python-multipart \
    soundfile \
    numpy

# Copy C binary from builder
COPY --from=builder /build/qwen_asr /app/qwen_asr

# Copy server code
COPY server.py .

# Default config for C backend
ENV INFERENCE_BACKEND=c \
    C_BINARY_PATH=/app/qwen_asr \
    MODEL_PATH=/model \
    HOST=0.0.0.0 \
    PORT=8765 \
    WORKERS=1 \
    SPEECH_RMS_THRESHOLD=0.01

EXPOSE 8765

# Model weights mounted at /model by the user
VOLUME /model

CMD ["python", "server.py"]
