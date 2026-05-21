# RunPod Serverless worker: faster-whisper on GPU with Firebase Admin.
# Built on RunPod's PyTorch base which already has CUDA + cuDNN configured.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    XDG_CACHE_HOME=/runpod-volume/.cache \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_DEVICE=cuda

# Fall back to writable /tmp paths when no network volume is attached.
# /runpod-volume is the standard mount when a network volume IS attached;
# without it, we need somewhere writable for HF model downloads.
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
# then cached on network volume / image layer for subsequent invocations).
# Don't pre-bake: a failed download here silently kills the image build.

CMD ["python", "-u", "handler.py"]
