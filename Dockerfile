# RunPod Serverless worker: faster-whisper on GPU with Firebase Admin.
#
# GPU/CUDA compatibility matrix:
#  - Image CUDA 12.4 (this file): works on Ada Lovelace (RTX 4090 sm_89),
#    Ampere (A6000/A100 sm_86/sm_80), Turing/Volta. The most-supported
#    sweet spot for RunPod RTX 4090 / 48 GB / 80 GB tiers.
#  - Image CUDA 12.9+: needed for Blackwell GPUs (RTX PRO 6000 sm_120, "PRO" tiers),
#    but the host NVIDIA driver on most current RunPod nodes is too old and
#    refuses to start a >=12.9 container (`nvidia-container-cli: requirement error`).
#
# If you ever switch the endpoint to a "PRO" / Blackwell GPU, bump this to
# runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204 (CUDA 12.8 — supports sm_120
# AND broadly available host drivers).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    XDG_CACHE_HOME=/runpod-volume/.cache \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_DEVICE=cuda

# Fall back to /tmp paths when no network volume is attached.
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

# Model downloads on first invocation (adds ~30-60s to the very first cold start).

CMD ["python", "-u", "handler.py"]
