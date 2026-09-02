"""Langfuse tracing setup.

Tracing is optional by design: if credentials are absent or Langfuse is
unreachable, the app must still answer questions. Observability that can take
down the service it observes is worse than no observability, and the Phase 3
eval runner needs to work in CI where Langfuse is not configured.

Set LANGFUSE_ENABLED=false to turn it off explicitly (the eval runner does this
to avoid flooding the free tier with a trace per golden-set question).
"""

import logging
import os

from langfuse import get_client

# Imported for its side effect: app.config loads .env into os.environ, which is
# where the Langfuse SDK looks for its credentials.
from app import config  # noqa: F401

logger = logging.getLogger(__name__)

_enabled: bool | None = None


def tracing_enabled() -> bool:
    """True when Langfuse is configured and reachable. Checked once."""
    global _enabled
    if _enabled is not None:
        return _enabled

    if os.getenv("LANGFUSE_ENABLED", "").lower() in ("false", "0", "no"):
        logger.info("Tracing disabled by LANGFUSE_ENABLED")
        _enabled = False
        return _enabled

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Tracing disabled: LANGFUSE_*_KEY not set")
        _enabled = False
        return _enabled

    try:
        _enabled = bool(get_client().auth_check())
        if not _enabled:
            logger.warning("Langfuse credentials rejected — tracing disabled")
    except Exception as exc:  # noqa: BLE001 — never let tracing break startup
        logger.warning("Langfuse unreachable (%s) — tracing disabled", exc)
        _enabled = False

    return _enabled


def flush() -> None:
    """Send buffered traces. Required before a short-lived process exits."""
    if tracing_enabled():
        try:
            get_client().flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse flush failed: %s", exc)
