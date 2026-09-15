"""YuE2 music generation on a single GPU, behind a small HTTP API.

Mirrors the minimax-music-gen contract shape so existing clients can be reused:
  POST /v1/audio/speech   {input=lyrics, instructions=style, seed, cot, abc,
                           cfg_scale, id, response_format} -> FLAC/WAV
  POST /v1/audio/plan     same body -> {id, abc, truncated, timing} (no audio)
  GET  /v1/models         model list
  GET  /health            503 until the pipeline is loaded

One job at a time (FIFO). Every generation also writes full artifacts
(score.abc, semantic.npy, latent.npy, audio.flac, request/config/result.json)
under --artifacts-dir/<id>/ for reproducibility and agentic editing.

Auth: if --api-key is set, /v1/* requires `Authorization: Bearer <key>` or
`X-API-Key: <key>`. Bind stays on 127.0.0.1; external access goes through
vast.ai's Caddy/Portal (TLS + auth) — defense in depth.
"""
import argparse
import io
import json
import queue
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_REPO = "m-a-p/YuE2-3B"
MODEL_REV = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"   # pinned per SPEC R5
VAE_REPO = "m-a-p/YuE2-Vae"
VAE_REV = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"     # pinned per SPEC R5
SAMPLE_RATE = 48_000
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")

STATE = {"ready": False, "status": "loading pipeline"}
QUEUE = queue.Queue()
PIPE = None
ARGS = None


class SpeechRequest(BaseModel):
    """Request body. `input`/`instructions` mirror the minimax contract;
    `cot`/`abc`/`cfg_scale`/`id` are YuE2-specific extensions."""

    input: str = Field(..., description="Lyrics; [Verse]/[Chorus] tags on their own lines.")
    instructions: str = Field(..., description="Style prompt: genre/BPM/key/vocal/arrangement.")
    seed: int | None = Field(None, ge=0, lt=2**63, description="Fixed seed reproduces a song; omitted = random, actual value returned in X-Seed.")
    cot: str = Field("full", description="full = melody+chord plan (default) | melody = cover-friendly | off = direct generation.")
    abc: str | None = Field(None, description="External ABC score; requires cot=full|melody.")
    cfg_scale: float | None = Field(None, ge=0, le=20, description="Text guidance; default 1.0 (full/melody) or 1.01 (off).")
    id: str | None = Field(None, description="Filename-safe id; artifacts land in <artifacts-dir>/<id>/.")
    response_format: str = Field("flac", description="flac (PCM_24, native) | wav (PCM_16).")


def _check_auth(authorization, x_api_key):
    if not ARGS.api_key:
        return
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    elif x_api_key:
        token = x_api_key
    if token != ARGS.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _validate(req: SpeechRequest):
    if req.cot not in ("full", "melody", "off"):
        raise HTTPException(422, "cot must be full, melody or off")
    if req.response_format not in ("flac", "wav"):
        raise HTTPException(422, "response_format must be flac or wav")
    if req.abc is not None and (req.cot == "off" or not req.abc.strip()):
        raise HTTPException(422, "abc requires nonempty text and cot=full|melody")
    if req.id is not None and (not ID_RE.fullmatch(req.id) or req.id in (".", "..")):
        raise HTTPException(422, "id must be filename-safe")


def _encode_audio(audio: np.ndarray, fmt: str):
    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    sf.write(buf, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_24")
    return buf.getvalue(), "audio/flac"


def _worker():
    """Loads the pipeline, then renders queued requests one at a time on the GPU."""
    global PIPE
    from yue2 import YuE2Pipeline

    try:
        PIPE = YuE2Pipeline.from_pretrained(
            MODEL_REPO, vae=VAE_REPO,
            revision=MODEL_REV, vae_revision=VAE_REV,
            memory_budget_gib=ARGS.memory_budget_gib,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001 — surface load failure via /health
        STATE["status"] = f"load failed: {exc}"
        return
    STATE["ready"] = True
    STATE["status"] = "ready"

    while True:
        job = QUEUE.get()
        try:
            req = job["request"]
            kwargs = dict(cot=req.cot, seed=job["seed"], id=job["id"])
            if req.abc is not None:
                kwargs["abc"] = req.abc
            if req.cfg_scale is not None:
                kwargs["cfg_scale"] = req.cfg_scale

            if job["kind"] == "plan":
                plan = PIPE.plan(style=req.instructions, lyrics=req.input, **kwargs)
                job["result"] = {
                    "id": job["id"],
                    "abc": plan.abc,
                    "truncated": plan.truncated,
                    "timing": plan.timing,
                }
            else:
                song = PIPE(style=req.instructions, lyrics=req.input, **kwargs)
                job["audio_seconds"] = len(song.audio) / song.sample_rate
                job["truncated"] = song.truncated
                job["request_id"] = song.request_identity
                job["result_bytes"], job["media_type"] = _encode_audio(
                    song.audio, req.response_format)
                try:
                    song.save_artifacts(str(Path(ARGS.artifacts_dir) / job["id"]))
                except Exception as exc:  # noqa: BLE001 — artifacts are best-effort
                    job["artifact_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 — report to caller
            job["error"] = exc
        finally:
            job["done"].set()
            QUEUE.task_done()


app = FastAPI(title="YuE2", description="Lyrics+style -> full song on one GPU (m-a-p/YuE2-3B).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    """503 until the model is on the GPU, so process supervisors can wait it out."""
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["status"])
    return {"status": "ready", "model": MODEL_REPO, "rev": MODEL_REV, "vae": VAE_REPO}


@app.get("/v1/models")
def list_models(authorization: str = Header(None), x_api_key: str = Header(None)):
    _check_auth(authorization, x_api_key)
    return {"object": "list", "data": [{"id": "yue2", "object": "model", "owned_by": "m-a-p"}]}


@app.post("/v1/audio/plan")
def plan(req: SpeechRequest, authorization: str = Header(None), x_api_key: str = Header(None)):
    _check_auth(authorization, x_api_key)
    _validate(req)
    seed = req.seed if req.seed is not None else random.randrange(2**63)
    song_id = req.id or f"song-{seed}"
    job = {"kind": "plan", "request": req, "seed": seed, "id": song_id,
           "done": threading.Event(), "result": None, "error": None}
    QUEUE.put(job)
    job["done"].wait()
    if job["error"] is not None:
        raise HTTPException(500, f"plan failed: {job['error']}")
    out = dict(job["result"])
    out["seed"] = seed
    return out


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest, authorization: str = Header(None), x_api_key: str = Header(None)):
    _check_auth(authorization, x_api_key)
    _validate(req)
    seed = req.seed if req.seed is not None else random.randrange(2**63)
    song_id = req.id or f"song-{seed}"
    job = {"kind": "song", "request": req, "seed": seed, "id": song_id,
           "done": threading.Event(), "result_bytes": None, "error": None}
    QUEUE.put(job)
    job["done"].wait()
    if job["error"] is not None:
        raise HTTPException(500, f"generation failed: {job['error']}")
    headers = {
        "X-Seed": str(seed),
        "X-Request-Id": job.get("request_id", ""),
        "X-Truncated": json.dumps(job["truncated"]),
    }
    if "artifact_error" in job:
        headers["X-Artifact-Error"] = job["artifact_error"][:200]
    return Response(job["result_bytes"], media_type=job["media_type"], headers=headers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--api-key", default=None, help="Bearer key required on /v1/* when set")
    ap.add_argument("--memory-budget-gib", type=float, default=24.0,
                    help="VRAM budget passed to YuE2Pipeline (<=12 selects smaller VAE chunks)")
    ap.add_argument("--artifacts-dir", default="/workspace/outputs",
                    help="Directory for save_artifacts() output per song id")
    ARGS = ap.parse_args()
    Path(ARGS.artifacts_dir).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")
