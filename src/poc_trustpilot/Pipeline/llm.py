"""Thin wrapper around the Claude API: structured-output calls, a JSONL cache and cost accounting."""
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from .config import (EFFORT, MODEL, PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_INPUT, PRICE_OUTPUT, ROOT)


class LLMError(RuntimeError):
    pass


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.cache_write_tokens += usage.cache_creation_input_tokens or 0
            self.cache_read_tokens += usage.cache_read_input_tokens or 0

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_INPUT + self.output_tokens * PRICE_OUTPUT
                + self.cache_write_tokens * PRICE_CACHE_WRITE + self.cache_read_tokens * PRICE_CACHE_READ) / 1e6

    def report(self) -> str:
        return (f"{self.calls} API calls | in {self.input_tokens:,} (cache write {self.cache_write_tokens:,}, "
                f"cache read {self.cache_read_tokens:,}) | out {self.output_tokens:,} | ~${self.cost_usd:.4f}")


METER = UsageMeter()
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv(ROOT / ".env")
        _client = anthropic.Anthropic()
    return _client


def parse(system: str, user: str, output_model: type[BaseModel], max_tokens: int = 4000) -> BaseModel:
    """One structured-output call. The system prompt is cached; a policy refusal falls back server-side."""
    try:
        resp = client().beta.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
            thinking={"type": "between_tools"},  # thinking off - this is a plain extraction task
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Authentication failed - set ANTHROPIC_API_KEY in your environment or in .env") from e
    METER.add(resp.usage)
    if resp.stop_reason == "refusal":
        raise LLMError(f"refused ({getattr(resp.stop_details, 'category', None)})")
    if resp.stop_reason == "max_tokens":
        raise LLMError("hit max_tokens before finishing the JSON")
    if resp.parsed_output is None:
        raise LLMError(f"no parsed output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


class JsonlCache:
    """Append-only key -> payload store, so re-runs never pay twice for the same input."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, payload: dict) -> None:
        rec = {"key": key, **payload}
        with self._lock:
            self.data[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
