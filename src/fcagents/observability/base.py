"""Backend-agnostic observability protocol, PII sanitizer, and zero-overhead NullTracer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable


def sanitize(value: Any) -> Any:
    """Recursively redact PII/secrets from strings, dicts, lists, and tuples before exporting telemetry."""
    if isinstance(value, str):
        from fcagents.guardrails.pii import redact_pii

        return redact_pii(value).redacted_text
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize(v) for v in value)
    return value


class ObservationHandle:
    """Thin, exception-safe handle around an underlying observation (or a no-op stub when ``obs`` is None)."""

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
            kwargs["output"] = sanitize(output)
        if metadata is not None:
            kwargs["metadata"] = sanitize(metadata)
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = sanitize(status_message)
        if kwargs:
            try:
                self._obs.update(**kwargs)
            except Exception:
                pass

    def score(self, *, name: str, value: float | int, comment: str | None = None) -> None:
        if self._obs is None:
            return
        try:
            clean_comment = sanitize(comment) if comment else None
            if hasattr(self._obs, "score"):
                self._obs.score(name=name, value=value, comment=clean_comment)
            elif hasattr(self._obs, "score_trace"):
                self._obs.score_trace(name=name, value=value, comment=clean_comment)
        except Exception:
            pass


@runtime_checkable
class Tracer(Protocol):
    """Pluggable observability backend interface."""

    name: str
    enabled: bool
    dashboard_url: str | None

    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[ObservationHandle]: ...

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ObservationHandle]: ...

    def flush(self) -> None: ...


class NullTracer:
    """Default no-op tracer used when no observability backend is configured."""

    name: str = "none"
    enabled: bool = False
    dashboard_url: str | None = None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[ObservationHandle]:
        yield ObservationHandle(None)

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ObservationHandle]:
        yield ObservationHandle(None)

    def flush(self) -> None:
        return None
