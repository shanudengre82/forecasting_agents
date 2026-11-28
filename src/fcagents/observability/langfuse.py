"""Optional Langfuse tracer backend (OTel-native SDK v3+). Only active when env vars & SDK are present."""

from __future__ import annotations

import atexit
import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fcagents.observability.base import ObservationHandle, sanitize


def langfuse_sdk_available() -> bool:
    """True when the optional ``langfuse`` package is installed."""
    return importlib.util.find_spec("langfuse") is not None


def langfuse_env_present() -> bool:
    """True when Langfuse credentials are configured and not explicitly disabled."""
    if os.environ.get("LANGFUSE_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip() and os.environ.get("LANGFUSE_SECRET_KEY", "").strip())


def langfuse_configured() -> bool:
    """True only when both Langfuse env vars and the optional ``langfuse`` SDK are available."""
    return langfuse_env_present() and langfuse_sdk_available()


def langfuse_dashboard_url() -> str | None:
    """Return the configured Langfuse dashboard URL when env vars are present, else None."""
    if not langfuse_env_present():
        return None
    url = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "").strip().rstrip("/")
    return url or "https://cloud.langfuse.com"


class LangfuseTracer:
    """Tracer implementation backed by the Langfuse Python SDK."""

    name: str = "langfuse"

    def __init__(self, client: Any | None = None):
        self._client = client
        self._initialized = client is not None

    @property
    def enabled(self) -> bool:
        return self._client is not None or langfuse_configured()

    @property
    def dashboard_url(self) -> str | None:
        if not self.enabled:
            return None
        return langfuse_dashboard_url()

    def _get_client(self):
        if self._initialized:
            return self._client
        self._initialized = True
        if not langfuse_configured():
            return None
        try:
            from langfuse import get_client

            self._client = get_client()
            atexit.register(self.flush)
        except Exception:
            self._client = None
        return self._client

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
        client = self._get_client()
        if client is None:
            yield ObservationHandle(None)
            return

        clean_input = sanitize(input) if input is not None else None
        clean_meta = sanitize(metadata) if metadata is not None else None
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
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ObservationHandle]:
        client = self._get_client()
        if client is None:
            yield ObservationHandle(None)
            return

        clean_input = sanitize(input) if input is not None else None
        clean_meta = sanitize(metadata) if metadata is not None else None
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

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
