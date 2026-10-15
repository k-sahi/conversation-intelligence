"""Thin LLM client with structured output via forced tool use.

We force the model to call a single tool whose `input_schema` is the Pydantic
JSON schema of the object we want. The tool input *is* the structured output,
so there is no free-text JSON to parse, and Pydantic validates it afterwards.

`FakeLLM` lets tests and offline demos exercise the full pipeline deterministically.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from ..config import settings

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class LLM(Protocol):
    model: str

    def structured(self, system: str, user: str, schema: type[T], tool_name: str = "emit",
                   max_tokens: int = 4096, temperature: float = 0.0) -> T: ...

    def text(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str: ...


class LLMError(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 180.0,
                 tracer: Any | None = None):
        self.model = model or settings.extraction_model
        self.api_key = api_key or settings.anthropic_api_key
        self.timeout = timeout
        self.tracer = tracer
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def _post(self, body: dict[str, Any], retries: int = 4) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last = None
        for attempt in range(retries):
            r = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(30, 2 ** attempt * 2))
                continue
            break
        raise LLMError(f"Anthropic API error {last}")

    def structured(self, system: str, user: str, schema: type[T], tool_name: str = "emit",
                   max_tokens: int = 4096, temperature: float = 0.0) -> T:
        tool = {
            "name": tool_name,
            "description": f"Return the result as a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for attempt in range(2):  # one repair round on validation failure
            body = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool_name},
                "messages": messages,
            }
            resp = self._post(body)
            block = next((b for b in resp.get("content", []) if b.get("type") == "tool_use"), None)
            if block is None:
                raise LLMError("model did not call the tool")
            try:
                return schema.model_validate(block["input"])
            except ValidationError as exc:
                if attempt == 1:
                    raise
                messages += [
                    {"role": "assistant", "content": resp["content"]},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": block["id"], "is_error": True,
                        "content": f"Schema validation failed, fix and call the tool again:\n{exc}",
                    }]},
                ]
        raise LLMError("unreachable")

    def text(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self._post({
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        })
        return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")

    def stream_text(self, system: str, user: str, max_tokens: int = 512):
        """Yield text deltas via SSE — used by the real-time latency benchmark."""
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": self.model, "max_tokens": max_tokens, "stream": True, "system": system,
                "messages": [{"role": "user", "content": user}]}
        with requests.post(ANTHROPIC_URL, headers=headers, json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:].strip())
                if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                    yield ev["delta"]["text"]


class FakeLLM:
    """Deterministic stand-in. `handler(system, user, schema)` returns a dict or str."""

    def __init__(self, handler: Callable[[str, str, type | None], Any], model: str = "fake"):
        self.handler = handler
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def structured(self, system, user, schema, tool_name="emit", max_tokens=4096, temperature=0.0):
        self.calls.append((system, user))
        return schema.model_validate(self.handler(system, user, schema))

    def text(self, system, user, max_tokens=1024, temperature=0.0):
        self.calls.append((system, user))
        return str(self.handler(system, user, None))
