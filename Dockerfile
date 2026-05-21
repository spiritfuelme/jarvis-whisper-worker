# RunPod Serverless worker: faster-whisper on GPU (CUDA 12.1) with Firebase Admin.
# Built on RunPod's PyTorch base which already has CUDA + cuDNN configured.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    XDG_CACHE_HOME=/runpod-volume/.cache \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py .

# Pre-warm the default model into the image so cold-start is faster.
# (You can comment this out if you want a smaller image; the model will
# download on first invocation into /runpod-volume/hf-cache instead.)
RUN python -c "from faster_whisper import WhisperModel; \
    WhisperModel('Systran/faster-distil-whisper-large-v3', device='cpu', compute_type='int8')"

CMD ["python", "-u", "handler.py"]
