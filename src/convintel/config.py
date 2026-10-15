"""Central settings, read from environment variables (and an optional .env file).

Every vendor key is optional at import time so that the pure-Python parts
(metrics, alignment, calibration) run and test without any credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so python-dotenv is not a hard dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- vendors -----------------------------------------------------------
    deepgram_api_key: str | None = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))
    deepgram_model: str = field(default_factory=lambda: _env("DEEPGRAM_MODEL", "nova-3"))
    hf_token: str | None = field(default_factory=lambda: _env("HF_TOKEN"))
    pyannote_pipeline: str = field(
        default_factory=lambda: _env("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
    )
    pyannote_device: str = field(default_factory=lambda: _env("PYANNOTE_DEVICE", "cpu"))
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    extraction_model: str = field(default_factory=lambda: _env("EXTRACTION_MODEL", "claude-sonnet-5-5"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "claude-opus-5-5"))
    realtime_llm_model: str = field(default_factory=lambda: _env("REALTIME_LLM_MODEL", "claude-haiku-5-5"))
    eleven_api_key: str | None = field(default_factory=lambda: _env("ELEVEN_API_KEY"))
    eleven_model: str = field(default_factory=lambda: _env("ELEVEN_MODEL", "eleven_v3_conversational"))
    eleven_voice_id: str = field(default_factory=lambda: _env("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"))
    livekit_url: str | None = field(default_factory=lambda: _env("LIVEKIT_URL"))
    livekit_api_key: str | None = field(default_factory=lambda: _env("LIVEKIT_API_KEY"))
    livekit_api_secret: str | None = field(default_factory=lambda: _env("LIVEKIT_API_SECRET"))
    arize_space_id: str | None = field(default_factory=lambda: _env("ARIZE_SPACE_ID"))
    arize_api_key: str | None = field(default_factory=lambda: _env("ARIZE_API_KEY"))
    arize_project: str = field(default_factory=lambda: _env("ARIZE_PROJECT", "conversation-intelligence"))

    # --- storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("CI_DATA_DIR", "data")))
    output_dir: Path = field(default_factory=lambda: Path(_env("CI_OUTPUT_DIR", "outputs")))

    # --- pipeline knobs ----------------------------------------------------
    utterance_gap_s: float = 0.8          # merge same-speaker words separated by less than this
    max_chunk_chars: int = 12_000         # transcript window per map-step LLM call
    chunk_overlap_utts: int = 6           # utterances repeated between windows


settings = Settings()
