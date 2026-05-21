# jarvis-whisper-worker

Container image for the [Jarvis transcription system](https://github.com/spiritfuelme).

Runs on **RunPod Serverless GPUs**. Pulls transcription jobs from a private Firestore
queue (`transcription_jobs` collection in the `ryan-chamberlin-brands` Firebase project),
transcribes the audio with `faster-whisper` using `distil-large-v3`, and uploads the
SRT/VTT/TXT/JSON outputs to Firebase Storage.

## Files
- `Dockerfile` — CUDA 12.4 + Python 3.11 + faster-whisper + firebase-admin
- `handler.py` — RunPod Serverless entry point
- `requirements.txt` — Python deps

## Runtime configuration

Set on the RunPod endpoint, **not** in this repo:
- `FIREBASE_SERVICE_ACCOUNT_JSON` — full JSON of a Firebase Admin SDK service account
- `FIREBASE_STORAGE_BUCKET` — defaults to `ryan-chamberlin-brands.firebasestorage.app`
- `WHISPER_COMPUTE_TYPE` — defaults to `float16`
- `WHISPER_DEVICE` — defaults to `cuda`

## Trigger model

The controller (`jarvis-jobs` CLI on the user's Mac) submits one POST request per job to:
```
https://api.runpod.ai/v2/{ENDPOINT_ID}/run
```
with payload:
```json
{ "input": { "job_id": "<firestore-doc-id>" } }
```

The handler atomically claims the job, processes it, and writes results back to
Firestore + Storage. See the controller code for the full flow.

This repo intentionally contains **no credentials** — service-account JSON is passed
via RunPod endpoint env vars at runtime.
