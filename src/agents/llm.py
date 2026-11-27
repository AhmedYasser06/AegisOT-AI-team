"""Provider-agnostic LLM resolution. A model is a LiteLLM model string; nothing is tied to one vendor.

crewAI agents and the LangGraph supervisor both resolve their model through `resolve_llm`.
Switch provider/model in .env (or configs/config.yaml); no code change.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Optional

from .roles import LLM_ROLES
from .settings import Settings

# Used only when LLM_PROVIDER is set but LLM_MODEL is not. Change freely.
# Model names get retired often (gemini-2.0-flash was shut down in 2026), so the Gemini default is Google's
# moving "latest flash" alias. For reproducible results set LLM_MODEL explicitly (list names: README section 4).
PROVIDER_DEFAULTS = {
    "gemini": "gemini/gemini-flash-lite-latest",
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-sonnet-4-6",
    "ollama": "ollama/llama3.1",
}
FALLBACK_MODEL = "gemini/gemini-flash-lite-latest"       # the team's default provider is Gemini


@dataclass(frozen=True)
class LLMSpec:
    model: str
    api_base: Optional[str]
    api_key: Optional[str]
    temperature: float
    timeout_s: int
    retries: int = 0                      # extra attempts on a temporary error (503 "high demand", 429, timeout)
    backoff_s: float = 1.5                # wait before retry n = backoff_s * n
    fallback_model: Optional[str] = None  # tried once when the main model keeps failing

    def public(self) -> dict:
        """Safe to expose over the API (no secrets)."""
        return {"model": self.model, "api_base": self.api_base, "temperature": self.temperature,
                "timeout_s": self.timeout_s, "api_key_set": bool(self.api_key),
                "retries": self.retries, "fallback_model": self.fallback_model}


def normalize_model(settings: Settings, raw: Optional[str]) -> str:
    provider = (settings.llm_provider or "").strip().lower()
    if raw and "/" in raw:
        return raw                                      # already "provider/model"
    if raw and provider:
        return f"{provider}/{raw}"                      # model name + LLM_PROVIDER
    if raw:
        return raw                                      # let LiteLLM infer (e.g. "gpt-4o-mini")
    if provider:
        return PROVIDER_DEFAULTS.get(provider, f"{provider}/default")
    return FALLBACK_MODEL


def resolve_llm(settings: Settings, role: str) -> LLMSpec:
    if role not in LLM_ROLES:
        raise ValueError(f"unknown role {role!r}")
    raw = getattr(settings, f"llm_model_{role}", None) or settings.llm_model
    return LLMSpec(model=normalize_model(settings, raw), api_base=settings.llm_api_base or None,
                   api_key=settings.llm_api_key or None, temperature=settings.llm_temperature,
                   timeout_s=settings.llm_timeout_s, retries=max(0, settings.llm_retries),
                   backoff_s=max(0.0, settings.llm_retry_backoff_s),
                   fallback_model=(normalize_model(settings, settings.llm_fallback_model)
                                   if settings.llm_fallback_model else None))


def _call_kwargs(spec: LLMSpec, system: str, user: str) -> dict:
    kwargs = dict(model=spec.model, temperature=spec.temperature, timeout=spec.timeout_s,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if spec.api_base:
        kwargs["api_base"] = spec.api_base
    if spec.api_key:
        kwargs["api_key"] = spec.api_key
    return kwargs


_TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|high demand|overloaded|"
                        r"ServiceUnavailable|RateLimit|Timeout|timed out|APIConnection|InternalServerError", re.I)


def is_transient(e: Exception) -> bool:
    """A temporary provider problem worth retrying (not a bad key / unknown model / invalid request)."""
    return bool(_TRANSIENT.search(f"{type(e).__name__} {e}"))


def complete_text(spec: LLMSpec, system: str, user: str) -> str:
    """Single chat completion through LiteLLM (specialist JSON answer and the non-streaming narrative).

    Temporary errors (e.g. Gemini 503 "model is currently experiencing high demand") are retried `spec.retries`
    times with a growing pause, then `spec.fallback_model` is tried once. Permanent errors are raised at once."""
    import litellm  # lazy: keeps rules-only mode and tests light

    def one(model_spec: LLMSpec) -> str:
        resp = litellm.completion(**_call_kwargs(model_spec, system, user))
        return (resp.choices[0].message.content or "").strip()

    last: Exception | None = None
    for attempt in range(spec.retries + 1):
        try:
            return one(spec)
        except Exception as e:
            last = e
            if not is_transient(e):
                raise
            if attempt < spec.retries:
                time.sleep(spec.backoff_s * (attempt + 1))
    if spec.fallback_model and spec.fallback_model != spec.model:
        try:
            return one(replace(spec, model=spec.fallback_model, retries=0, fallback_model=None))
        except Exception:
            pass                                    # report the ORIGINAL error: it is the one to fix
    assert last is not None
    raise last


def stream_text(spec: LLMSpec, system: str, user: str):
    """Same call, but yields the answer piece by piece (token streaming for the dashboard)."""
    import litellm

    for chunk in litellm.completion(**_call_kwargs(spec, system, user), stream=True):
        piece = chunk.choices[0].delta.content if chunk.choices else None
        if piece:
            yield piece


def iter_with_deadline(pieces, deadline_s: float):
    """Yield from `pieces`, but give up (TimeoutError) once `deadline_s` seconds have passed IN TOTAL.

    The per-request LLM timeout does not bound a stream that stalls halfway (we saw one hang for 4 minutes).
    The source is read in a helper thread; when the deadline passes we stop waiting and the caller falls back to
    the template. The helper thread cannot be killed: it ends when the provider call ends, and its output is dropped.
    """
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def work():
        try:
            for piece in pieces:
                if stop.is_set():
                    break
                q.put(("piece", piece))
            q.put(("end", None))
        except Exception as e:                      # handed to the consumer below
            q.put(("error", e))

    threading.Thread(target=work, daemon=True, name="llm-stream").start()
    end = time.monotonic() + deadline_s
    try:
        while True:
            try:
                kind, value = q.get(timeout=max(0.0, end - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"LLM stream exceeded the {deadline_s:g}s deadline") from None
            if kind == "piece":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        stop.set()


# ───────── error text for the `notes` field ─────────
_SECRET = re.compile(r"(key=)[^&\s'\"]+|AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{16,}", re.I)


def short_error(e: Exception, limit: int = 180) -> str:
    """Short, secret-free description of an exception for the `notes` field."""
    msg = " ".join(str(e).split())
    if re.search(r"429|RESOURCE_EXHAUSTED|RateLimit|quota", msg, re.I):
        return (f"{type(e).__name__}: rate limit / quota exceeded (HTTP 429). Free tiers allow only a few requests per "
                f"minute per model: use different models per agent, LLM_AGENTS, LLM_MAX_CONCURRENT=1, or a paid key")
    msg = _SECRET.sub(lambda m: f"{m.group(1)}***" if m.group(1) else "***", msg)
    return f"{type(e).__name__}: {msg[:limit]}"
