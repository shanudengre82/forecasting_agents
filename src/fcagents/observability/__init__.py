"""Backend-agnostic observability facade.

Defaults to ``NullTracer`` (zero overhead, no UI dashboard) unless an observability backend
is explicitly registered via ``set_tracer(...)`` or configured via environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fcagents.observability.base import NullTracer, ObservationHandle, Tracer, sanitize

_NULL = NullTracer()
_CUSTOM_TRACER: Tracer | None = None
_AUTO_TRACER: Tracer | None = None


def set_tracer(tracer: Tracer | None) -> None:
    """Register a custom ``Tracer`` backend (or pass ``None`` to restore auto-detection)."""
    global _CUSTOM_TRACER
    _CUSTOM_TRACER = tracer


def reset_tracer() -> None:
    """Clear any cached or custom tracer (useful in tests when toggling environment variables)."""
    global _CUSTOM_TRACER, _AUTO_TRACER
    _CUSTOM_TRACER = None
    _AUTO_TRACER = None


def get_tracer() -> Tracer:
    """Return the active ``Tracer`` backend, or ``NullTracer`` when none is configured."""
    if _CUSTOM_TRACER is not None:
        return _CUSTOM_TRACER

    backend = os.environ.get("TRACING_BACKEND", "auto").strip().lower()
    if backend in {"none", "off", "0", "false"}:
        return _NULL

    global _AUTO_TRACER
    if backend in {"auto", "langfuse"}:
        from fcagents.observability.langfuse import LangfuseTracer, langfuse_configured

        if langfuse_configured():
            if _AUTO_TRACER is None or _AUTO_TRACER.name != "langfuse":
                _AUTO_TRACER = LangfuseTracer()
            return _AUTO_TRACER

    return _NULL


def is_enabled() -> bool:
    """True when an active observability backend is configured and ready."""
    return bool(get_tracer().enabled)


def tracer_name() -> str:
    """Name of the active tracer backend (e.g. ``'none'``, ``'langfuse'``, or custom)."""
    return get_tracer().name


def dashboard_url() -> str | None:
    """Dashboard URL for the active tracer backend, or ``None`` when unconfigured."""
    tracer = get_tracer()
    if not tracer.enabled:
        return None
    override = os.environ.get("TRACING_DASHBOARD_URL", "").strip()
    return override or tracer.dashboard_url


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[ObservationHandle]:
    with get_tracer().span(name, input=input, metadata=metadata, session_id=session_id, tags=tags) as obs:
        yield obs


@contextmanager
def trace_generation(
    name: str,
    *,
    model: str,
    input: Any = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[ObservationHandle]:
    with get_tracer().generation(
        name,
        model=model,
        input=input,
        model_parameters=model_parameters,
        metadata=metadata,
    ) as obs:
        yield obs


def flush() -> None:
    get_tracer().flush()


__all__ = [
    "NullTracer",
    "ObservationHandle",
    "Tracer",
    "dashboard_url",
    "flush",
    "get_tracer",
    "is_enabled",
    "reset_tracer",
    "sanitize",
    "set_tracer",
    "trace_generation",
    "trace_span",
    "tracer_name",
]
