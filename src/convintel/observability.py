"""Tracing to Arize via OpenTelemetry + OpenInference semantic conventions.

If `arize-otel` / opentelemetry are not installed, or ARIZE_* keys are unset,
every call becomes a no-op so the pipeline runs unchanged.

Each batch job is one trace:
  pipeline (CHAIN)
    ├── asr.deepgram (CHAIN)            attrs: model, audio_seconds, n_words
    ├── diarization.pyannote (CHAIN)    attrs: n_speakers, overlap_ratio
    ├── align (CHAIN)
    └── extract (CHAIN)
          ├── llm.map[i] (LLM)          attrs: llm.model_name, input/output
          └── llm.reduce (LLM)
Eval scores are attached to the root span as `eval.*` attributes, so in Arize you
can filter traces by e.g. `eval.judge.faithfulness < 3`.
"""
from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator

from .config import settings

_tracer = None
_initialised = False


def init_tracing(project: str | None = None):
    global _tracer, _initialised
    if _initialised:
        return _tracer
    _initialised = True
    if not (settings.arize_space_id and settings.arize_api_key):
        return None
    try:
        from arize.otel import register
        from opentelemetry import trace

        provider = register(space_id=settings.arize_space_id, api_key=settings.arize_api_key,
                            project_name=project or settings.arize_project)
        _tracer = trace.get_tracer("convintel", tracer_provider=provider)
    except Exception:  # pragma: no cover - optional dependency
        _tracer = None
    return _tracer


class _NoopSpan:
    def set_attribute(self, *_: Any) -> None: ...
    def set_attributes(self, *_: Any) -> None: ...
    def record_exception(self, *_: Any) -> None: ...


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", **attrs: Any) -> Iterator[Any]:
    tracer = init_tracing()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for k, v in attrs.items():
            s.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            raise
