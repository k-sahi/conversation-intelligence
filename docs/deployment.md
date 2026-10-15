# Deployment

Two containers, deployed independently because they scale on different axes.

| Image | Contents | Scales on | Where |
|---|---|---|---|
| `Dockerfile.api` | FastAPI + batch pipeline + pyannote (torch CPU wheels) | uploads, GPU-minutes | any container host; GPU node for faster diarization |
| `Dockerfile.agent` | LiveKit Agents worker, no torch | concurrent rooms | LiveKit Cloud (`lk agent create`) or self-hosted |

## Local

```bash
cp .env.example .env     # fill keys
docker compose up --build
curl localhost:8000/healthz
```

## Agent on LiveKit Cloud

```bash
lk cloud auth                              # link the CLI to your project
lk agent create                            # first deploy: builds Dockerfile.agent, writes ids into livekit.toml
lk agent update-secrets --secrets-file .env
lk agent deploy                            # later deploys
```
Rooms created with the token from `POST /v1/livekit/token` dispatch the `meeting-notes` agent automatically.
The agent writes outputs to `CI_OUTPUT_DIR`; in Cloud this is ephemeral, so for persistence upload
`record.json` to object storage in the shutdown callback (S3/GCS client of your choice).

## API in production

* Put uploads behind object storage + signed URLs instead of streaming through the API for large files.
* Replace `BackgroundTasks` with a queue (Arq/Celery/Cloud Tasks); run pyannote workers on GPU
  (`PYANNOTE_DEVICE=cuda`) — diarization dominates batch wall time on CPU.
* Mount a persistent volume at `/models` so pyannote weights are downloaded once.
* Run one uvicorn worker per container (the pipeline is CPU/GPU-bound), scale by replicas.

## Secrets

All keys come from environment variables; nothing is read from the repo. Rotate per environment.
`ARIZE_*` unset ⇒ tracing is a no-op.

## Privacy

Meeting audio is personal data. pyannote runs in our container (no third party); Deepgram, Anthropic and
ElevenLabs receive audio/text — check each vendor's data-retention settings and your consent flow.
Delete raw audio after processing if it isn't needed (`outputs/<id>/audio.*`).
