"""Langfuse observability wrapper (OTel-native SDK v3/v4) with automatic zero-overhead no-op fallback."""

from __future__ import annotations

import atexit
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fcagents.guardrails.pii import redact_pii

_CLIENT: Any = None
_CLIENT_INITIALIZED = False


def is_enabled() -> bool:
    """True when Langfuse credentials are configured and tracing is not explicitly disabled."""
    if os.environ.get("LANGFUSE_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip() and os.environ.get("LANGFUSE_SECRET_KEY", "").strip())


def _get_client():
    global _CLIENT, _CLIENT_INITIALIZED
    if not is_enabled():
        return None
    if _CLIENT_INITIALIZED:
        return _CLIENT
    _CLIENT_INITIALIZED = True
    try:
        from langfuse import get_client

        _CLIENT = get_client()
        atexit.register(flush)
    except Exception:
        _CLIENT = None
    return _CLIENT


def reset_client() -> None:
    """Reset the cached client (used in unit tests when toggling env vars or mocks)."""
    global _CLIENT, _CLIENT_INITIALIZED
    _CLIENT = None
    _CLIENT_INITIALIZED = False


def _sanitize(value: Any) -> Any:
    """Recursively redact PII from strings/dicts/lists before sending to Langfuse."""
    if isinstance(value, str):
        return redact_pii(value).redacted_text
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(v) for v in value)
    return value


class ObservationHandle:
    """Thin, exception-safe wrapper around a Langfuse observation (or a no-op stub)."""

    def __init__(self, obs: Any = None):
        self._obs = obs

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._obs is None:
            return
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = _sanitize(output)
        if metadata is not None:
            kwargs["metadata"] = _sanitize(metadata)
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = _sanitize(status_message)
        if kwargs:
            try:
                self._obs.update(**kwargs)
            except Exception:
                pass

    def score(self, *, name: str, value: float | int, comment: str | None = None) -> None:
        if self._obs is None:
            return
        try:
            if hasattr(self._obs, "score"):
                self._obs.score(name=name, value=value, comment=_sanitize(comment) if comment else None)
            elif hasattr(self._obs, "score_trace"):
                self._obs.score_trace(name=name, value=value, comment=_sanitize(comment) if comment else None)
        except Exception:
            pass


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[ObservationHandle]:
    """Open a Langfuse span (and propagate ``session_id``/``tags`` when provided), or no-op if disabled."""
    client = _get_client()
    if client is None:
        yield ObservationHandle(None)
        return

    clean_input = _sanitize(input) if input is not None else None
    clean_meta = _sanitize(metadata) if metadata is not None else None
    try:
        span_cm = client.start_as_current_observation(
            as_type="span",
            name=name,
            input=clean_input,
            metadata=clean_meta,
        )
    except Exception:
        yield ObservationHandle(None)
        return

    with span_cm as obs:
        if session_id or tags:
            try:
                from langfuse import propagate_attributes

                prop_kwargs: dict[str, Any] = {}
                if session_id:
                    prop_kwargs["session_id"] = session_id
                if tags:
                    prop_kwargs["tags"] = tags
                with propagate_attributes(**prop_kwargs):
                    yield ObservationHandle(obs)
                return
            except Exception:
                pass
        yield ObservationHandle(obs)


@contextmanager
def trace_generation(
    name: str,
    *,
    model: str,
    input: Any = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[ObservationHandle]:
    """Open a Langfuse ``generation`` observation for an LLM call, or no-op if disabled."""
    client = _get_client()
    if client is None:
        yield ObservationHandle(None)
        return

    clean_input = _sanitize(input) if input is not None else None
    clean_meta = _sanitize(metadata) if metadata is not None else None
    try:
        gen_cm = client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            model_parameters=model_parameters,
            input=clean_input,
            metadata=clean_meta,
        )
    except Exception:
        yield ObservationHandle(None)
        return

    with gen_cm as obs:
        yield ObservationHandle(obs)


def flush() -> None:
    """Flush buffered Langfuse traces (safe no-op when disabled)."""
    if _CLIENT is not None:
        try:
            _CLIENT.flush()
        except Exception:
            pass
