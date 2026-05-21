# RunPod Serverless worker: faster-whisper on GPU with Firebase Admin.
#
# Base: CUDA 12.9 + PyTorch 2.9.1 — needed for Blackwell GPUs (sm_120, e.g.
# RTX PRO 6000 Blackwell on RunPod's "24 GB PRO" tier). PyTorch 2.4 / CUDA 12.4
# only ships kernels for sm_50..sm_90 and fails RunPod's CUDA fitness check on
# Blackwell hardware ("no kernel image is available for execution on the device").
#
# Note: faster-whisper uses CTranslate2, not PyTorch, for the actual inference.
# PyTorch needs to be present + functional only because RunPod's worker startup
# fitness check imports it and verifies torch.cuda is healthy before serving.

FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    XDG_CACHE_HOME=/runpod-volume/.cache \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_DEVICE=cuda

# Fall back to writable /tmp paths when no network volume is attached.
RUN mkdir -p /runpod-volume/hf-cache /runpod-volume/.cache || \
    (mkdir -p /tmp/hf-cache /tmp/.cache && \
     echo "HF_HOME=/tmp/hf-cache" >> /etc/environment && \
     echo "XDG_CACHE_HOME=/tmp/.cache" >> /etc/environment)

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py .

# Model downloads on first invocation (adds ~30-60s to the very first cold start,
# then cached on the network volume / image layer for subsequent invocations).

CMD ["python", "-u", "handler.py"]
