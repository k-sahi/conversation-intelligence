"""FastAPI service for the batch path + helpers for the real-time path.

Endpoints
  GET  /healthz                         liveness
  POST /v1/meetings                     upload audio (multipart) -> job id; processing runs in background
  GET  /v1/meetings/{id}                job status + timings
  GET  /v1/meetings/{id}/transcript     diarized transcript (JSON)
  GET  /v1/meetings/{id}/record         structured record (JSON) or ?format=md
  POST /v1/meetings/{id}/recap-audio    ElevenLabs speech of the summary (audio/mpeg)
  POST /v1/livekit/token                join token for a room (dispatches the note-taker agent)

Jobs are kept in a small JSON-on-disk store so the container can restart without losing
results. For multi-replica production swap `JobStore` for Redis/Postgres and the
BackgroundTasks runner for a queue (Celery/Arq) — the interface is the same.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from ..config import settings
from ..pipeline import BatchPipeline, PipelineOptions, render_markdown
from ..schemas import MeetingRecord

app = FastAPI(title="Conversation Intelligence API", version="0.1.0")
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB
ALLOWED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _p(self, jid: str) -> Path:
        if not jid.replace("-", "").isalnum():
            raise HTTPException(400, "bad id")
        return self.root / jid / "job.json"

    def put(self, jid: str, **fields) -> dict:
        with self._lock:
            p = self._p(jid)
            p.parent.mkdir(parents=True, exist_ok=True)
            cur = json.loads(p.read_text()) if p.exists() else {"id": jid}
            cur.update(fields, updated_at=time.time())
            p.write_text(json.dumps(cur))
            return cur

    def get(self, jid: str) -> dict:
        p = self._p(jid)
        if not p.exists():
            raise HTTPException(404, "meeting not found")
        return json.loads(p.read_text())


store = JobStore(settings.output_dir)
_pipeline: BatchPipeline | None = None


def pipeline() -> BatchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BatchPipeline(output_dir=settings.output_dir)
    return _pipeline


class JobOut(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    timings: dict | None = None
    error: str | None = None


def _process(jid: str, audio: Path, opts: PipelineOptions) -> None:
    store.put(jid, status="running")
    try:
        res = pipeline().run(audio, meeting_id=jid, opts=opts)
        store.put(jid, status="done", timings=res.timings)
    except Exception as exc:  # surface the error to the client instead of a silent 500 later
        store.put(jid, status="failed", error=f"{type(exc).__name__}: {exc}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "deepgram": bool(settings.deepgram_api_key), "hf": bool(settings.hf_token),
            "anthropic": bool(settings.anthropic_api_key), "eleven": bool(settings.eleven_api_key)}


@app.post("/v1/meetings", response_model=JobOut, status_code=202)
async def create_meeting(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    multichannel: bool = Form(False),
    num_speakers: int | None = Form(None),
    keyterms: str | None = Form(None, description="comma-separated domain terms"),
):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"unsupported audio type {suffix}")
    jid = uuid.uuid4().hex[:12]
    dest = settings.output_dir / jid / f"audio{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "file too large")
            fh.write(chunk)
    opts = PipelineOptions(multichannel=multichannel, num_speakers=num_speakers,
                           keyterms=[k.strip() for k in keyterms.split(",")] if keyterms else None)
    store.put(jid, status="queued", filename=file.filename)
    background.add_task(_process, jid, dest, opts)
    return JobOut(id=jid, status="queued")


@app.get("/v1/meetings/{jid}", response_model=JobOut)
def get_meeting(jid: str):
    j = store.get(jid)
    return JobOut(id=jid, status=j["status"], timings=j.get("timings"), error=j.get("error"))


def _artifact(jid: str, name: str) -> Path:
    j = store.get(jid)
    p = settings.output_dir / jid / name
    if not p.exists():
        raise HTTPException(409 if j["status"] != "done" else 404, f"{name} not ready (status={j['status']})")
    return p


@app.get("/v1/meetings/{jid}/transcript")
def get_transcript(jid: str):
    return json.loads(_artifact(jid, "transcript.json").read_text())


@app.get("/v1/meetings/{jid}/record")
def get_record(jid: str, format: Literal["json", "md"] = "json"):
    rec = MeetingRecord.model_validate_json(_artifact(jid, "record.json").read_text())
    if format == "md":
        return PlainTextResponse(render_markdown(rec), media_type="text/markdown")
    return rec


@app.post("/v1/meetings/{jid}/recap-audio")
def recap_audio(jid: str):
    if not settings.eleven_api_key:
        raise HTTPException(503, "ELEVEN_API_KEY not configured")
    rec = MeetingRecord.model_validate_json(_artifact(jid, "record.json").read_text())
    text = rec.summary
    if rec.action_items:
        text += " Action items: " + "; ".join(
            f"{a.owner + ' will ' if a.owner else ''}{a.task}" for a in rec.action_items[:5]) + "."
    cache = settings.output_dir / jid / "recap.mp3"
    if not cache.exists():
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{settings.eleven_voice_id}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": settings.eleven_api_key, "content-type": "application/json"},
            json={"text": text[:4500], "model_id": settings.eleven_model}, timeout=120)
        if r.status_code != 200:
            raise HTTPException(502, f"ElevenLabs error {r.status_code}: {r.text[:200]}")
        cache.write_bytes(r.content)
    return Response(cache.read_bytes(), media_type="audio/mpeg")


class TokenIn(BaseModel):
    room: str
    identity: str
    name: str | None = None


@app.post("/v1/livekit/token")
def livekit_token(body: TokenIn):
    if not (settings.livekit_api_key and settings.livekit_api_secret and settings.livekit_url):
        raise HTTPException(503, "LiveKit credentials not configured")
    from livekit import api as lk_api

    grants = lk_api.VideoGrants(room_join=True, room=body.room, can_publish=True, can_subscribe=True)
    token = (lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
             .with_identity(body.identity).with_name(body.name or body.identity).with_grants(grants)
             .with_room_config(lk_api.RoomConfiguration(
                 agents=[lk_api.RoomAgentDispatch(agent_name="meeting-notes")])))
    return {"url": settings.livekit_url, "token": token.to_jwt()}
