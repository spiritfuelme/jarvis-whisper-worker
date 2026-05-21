"""
RunPod Serverless handler for Whisper transcription jobs.

Trigger model: caller (our controller CLI) sends one request per job with:
    { "input": { "job_id": "<firestore-doc-id>" } }

The handler:
  1. Loads the Firestore doc for that job.
  2. Atomically claims it (status pending -> processing).
  3. Downloads the audio from `audio_url` OR `audio_storage_path`.
  4. Transcribes with faster-whisper using model from job (default distil-large-v3).
  5. Uploads .srt .vtt .txt .json to Storage under the path the job specifies.
  6. Updates the job doc with status=done + output URLs + timing.

Failures bump `attempts`; if `attempts >= max_attempts` it's marked failed.

IMPORTANT: All initialization (Firebase, models) is LAZY. If env vars are missing
or invalid we return a clear error from the handler instead of crashing the
worker container on startup (which is invisible to RunPod and shows as
"unhealthy worker").
"""

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import runpod
import requests

# --- Worker identity (no external deps, safe at import) --------------------

WORKER_ID = f"runpod-{os.environ.get('RUNPOD_POD_ID', 'local')}-{uuid.uuid4().hex[:8]}"

def _scratch_root() -> Path:
    """Pick a writable scratch dir. /runpod-volume is only present if a
    network volume is attached to the endpoint; fall back to /tmp."""
    for candidate in (os.environ.get("SCRATCH_DIR"), "/runpod-volume/scratch", "/tmp/scratch"):
        if not candidate:
            continue
        try:
            p = Path(candidate)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception as e:
            print(f"[handler] scratch candidate {candidate} unusable: {e}", flush=True)
    return Path("/tmp")

SCRATCH = _scratch_root()
print(f"[handler] worker boot OK. WORKER_ID={WORKER_ID} SCRATCH={SCRATCH}", flush=True)

# Firebase / model handles, populated lazily -------------------------------

_FB = {"db": None, "bucket": None, "ok": False}
_MODELS = {}

def _ensure_firebase():
    if _FB["ok"]:
        return _FB["db"], _FB["bucket"]
    import firebase_admin
    from firebase_admin import credentials, firestore, storage

    if firebase_admin._apps:
        _FB["db"] = firestore.client()
        _FB["bucket"] = storage.bucket()
        _FB["ok"] = True
        return _FB["db"], _FB["bucket"]

    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "ryan-chamberlin-brands.firebasestorage.app")
    cred_json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")

    if cred_json_str:
        cred_json_str = cred_json_str.strip()
        try:
            cred_dict = json.loads(cred_json_str)
        except Exception as e:
            raise RuntimeError(
                f"FIREBASE_SERVICE_ACCOUNT_JSON env var is not valid JSON: "
                f"{e.__class__.__name__}: {e} "
                f"(length={len(cred_json_str)} starts_with={cred_json_str[:30]!r})"
            )
        cred = credentials.Certificate(cred_dict)
    elif cred_path and Path(cred_path).exists():
        cred = credentials.Certificate(cred_path)
    else:
        raise RuntimeError(
            "No Firebase credentials. Set FIREBASE_SERVICE_ACCOUNT_JSON env var on the endpoint."
        )

    firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    _FB["db"] = firestore.client()
    _FB["bucket"] = storage.bucket()
    _FB["ok"] = True
    print(f"[handler] firebase initialized; bucket={_FB['bucket'].name}", flush=True)
    return _FB["db"], _FB["bucket"]

def _get_model(name: str):
    if name in _MODELS:
        return _MODELS[name]
    from faster_whisper import WhisperModel
    name_map = {
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "large-v3":         "Systran/faster-whisper-large-v3",
        "medium.en":        "Systran/faster-whisper-medium.en",
        "base.en":          "Systran/faster-whisper-base.en",
    }
    repo = name_map.get(name, name)
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    print(f"[handler] loading model {repo} on {device}/{compute_type}", flush=True)
    model = WhisperModel(repo, device=device, compute_type=compute_type)
    _MODELS[name] = model
    return model

def _claim_job(ref):
    from firebase_admin import firestore as fs_mod
    from google.cloud.firestore import SERVER_TIMESTAMP

    db, _ = _ensure_firebase()

    @fs_mod.transactional
    def _txn(tx, ref):
        snap = ref.get(transaction=tx)
        if not snap.exists:
            raise RuntimeError("job doc does not exist")
        data = snap.to_dict()
        status = data.get("status", "pending")
        attempts = int(data.get("attempts", 0))
        max_attempts = int(data.get("max_attempts", 3))
        if status == "done": return None
        if status == "processing" and data.get("worker_id") != WORKER_ID: return None
        if attempts >= max_attempts and status != "retry": return None
        tx.update(ref, {
            "status": "processing",
            "worker_id": WORKER_ID,
            "claimed_at": SERVER_TIMESTAMP,
            "started_at": SERVER_TIMESTAMP,
            "attempts": attempts + 1,
            "updated_at": SERVER_TIMESTAMP,
            "error": None,
        })
        data["attempts"] = attempts + 1
        return data

    tx = db.transaction()
    return _txn(tx, ref)

def _download_audio(job: dict, dest: Path) -> Path:
    _, bucket = _ensure_firebase()
    if job.get("audio_url"):
        url = job["audio_url"]
        print(f"[handler] downloading {url[:120]}", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return dest
    if job.get("audio_storage_path"):
        blob = bucket.blob(job["audio_storage_path"])
        blob.download_to_filename(str(dest))
        return dest
    raise ValueError("job has neither audio_url nor audio_storage_path")

def _fmt_srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _write_outputs(segments, out_prefix: Path, info):
    srt_lines, vtt_lines, txt_lines, json_segments = [], ["WEBVTT", ""], [], []
    for i, seg in enumerate(segments, 1):
        text = (seg.text or "").strip()
        if not text: continue
        start_srt = _fmt_srt_ts(seg.start)
        end_srt = _fmt_srt_ts(seg.end)
        start_vtt = start_srt.replace(",", ".")
        end_vtt = end_srt.replace(",", ".")
        srt_lines += [str(i), f"{start_srt} --> {end_srt}", text, ""]
        vtt_lines += [f"{start_vtt} --> {end_vtt}", text, ""]
        txt_lines.append(text)
        json_segments.append({
            "id": i, "start": seg.start, "end": seg.end, "text": text,
            "words": [
                {"start": w.start, "end": w.end, "word": w.word, "probability": getattr(w, "probability", None)}
                for w in (seg.words or [])
            ] if seg.words else [],
        })
    out_prefix.with_suffix(".srt").write_text("\n".join(srt_lines))
    out_prefix.with_suffix(".vtt").write_text("\n".join(vtt_lines))
    out_prefix.with_suffix(".txt").write_text(" ".join(txt_lines))
    out_prefix.with_suffix(".json").write_text(json.dumps({
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": json_segments,
    }, ensure_ascii=False))
    return {ext: str(out_prefix.with_suffix(f".{ext}")) for ext in ("srt", "vtt", "txt", "json")}

def _upload_outputs(local_outputs: dict, storage_prefix: str) -> dict:
    _, bucket = _ensure_firebase()
    out_urls = {}
    for ext, local_path in local_outputs.items():
        blob_path = f"{storage_prefix}.{ext}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, content_type={
            "srt": "application/x-subrip",
            "vtt": "text/vtt",
            "txt": "text/plain; charset=utf-8",
            "json": "application/json",
        }[ext])
        out_urls[ext] = f"gs://{bucket.name}/{blob_path}"
    return out_urls

def handler(event):
    from google.cloud.firestore import SERVER_TIMESTAMP

    payload = (event or {}).get("input") or {}

    # Diagnostic ping: {"input": {"diag": true}} → returns worker env state
    if payload.get("diag"):
        info = {
            "worker_id": WORKER_ID,
            "scratch": str(SCRATCH),
            "python": sys.version.split()[0],
            "env": {
                "FIREBASE_STORAGE_BUCKET": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
                "FIREBASE_SERVICE_ACCOUNT_JSON_len": len(os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON","")),
                "HF_HOME": os.environ.get("HF_HOME",""),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",""),
            },
        }
        try:
            _ensure_firebase()
            info["firebase"] = "OK"
        except Exception as e:
            info["firebase"] = f"FAIL: {e.__class__.__name__}: {e}"
        return info

    job_id = payload.get("job_id")
    if not job_id:
        return {"error": "missing input.job_id (or input.diag=true)"}

    try:
        db, _ = _ensure_firebase()
    except Exception as e:
        return {"error": f"firebase init failed: {e}", "job_id": job_id}

    ref = db.collection("transcription_jobs").document(job_id)
    try:
        job = _claim_job(ref)
    except Exception as e:
        return {"error": f"claim failed: {e}", "job_id": job_id}
    if not job:
        return {"job_id": job_id, "skipped": True, "reason": "already done / claimed"}

    model_name = job.get("model", "distil-large-v3")
    language = job.get("language") or "en"
    storage_prefix = job.get("storage_prefix") or f"transcripts/{job_id}"

    work = SCRATCH / job_id
    work.mkdir(parents=True, exist_ok=True)
    audio = work / "audio.bin"
    out_prefix = work / "out"

    t0 = time.time()
    try:
        _download_audio(job, audio)
        dl = time.time() - t0

        model = _get_model(model_name)
        t1 = time.time()
        segments, info = model.transcribe(
            str(audio),
            language=language,
            beam_size=int(job.get("beam_size", 5)),
            vad_filter=bool(job.get("vad_filter", True)),
            word_timestamps=True,
        )
        segments = list(segments)
        infer = time.time() - t1

        outputs = _write_outputs(segments, out_prefix, info)
        urls = _upload_outputs(outputs, storage_prefix)
        upload = time.time() - t0 - dl - infer

        ref.update({
            "status": "done",
            "finished_at": SERVER_TIMESTAMP,
            "updated_at": SERVER_TIMESTAMP,
            "duration_sec": info.duration,
            "detected_language": info.language,
            "detected_language_probability": info.language_probability,
            "transcripts": urls,
            "timing": {
                "download_sec": round(dl, 2),
                "inference_sec": round(infer, 2),
                "upload_sec": round(upload, 2),
                "total_sec": round(time.time() - t0, 2),
                "realtime_factor": round(info.duration / max(infer, 1e-6), 2),
            },
            "worker_id": WORKER_ID,
            "error": None,
        })
        return {"job_id": job_id, "ok": True, "duration_sec": info.duration,
                "realtime_factor": round(info.duration / max(infer, 1e-6), 2),
                "transcripts": urls}
    except Exception as e:
        err = f"{e.__class__.__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        snap = ref.get().to_dict() or {}
        attempts = int(snap.get("attempts", 0))
        max_attempts = int(snap.get("max_attempts", 3))
        new_status = "failed" if attempts >= max_attempts else "retry"
        ref.update({
            "status": new_status,
            "error": err[-1500:],
            "updated_at": SERVER_TIMESTAMP,
        })
        return {"job_id": job_id, "ok": False, "status": new_status, "error": err}
    finally:
        for p in work.glob("*"):
            try: p.unlink()
            except: pass

runpod.serverless.start({"handler": handler})
