"""Shared model-call helper for the LLM-led experiment.

Both LLM roles (preference brain + decision actor) reach the model through the
same OpenAI-style ``model_chat_completion``. They must share ONE transient-error
policy — retry a few times with backoff before degrading — rather than the
decision side retrying while the preference side fell straight through on the
first timeout (which silently dropped a whole step's ledger update).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .log_color import _line_end, _tag

MODEL_CALL_ATTEMPTS = 3


def load_content(resp: Any) -> str:
    """Extract the assistant message content from a chat-completion response."""
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("model response missing choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response content empty")
    return content


def call_content_with_retry(
    api: Any,
    payload: dict[str, Any],
    log: logging.Logger,
    *,
    attempts: int = MODEL_CALL_ATTEMPTS,
) -> str:
    """Call ``model_chat_completion`` and return its content string, retrying
    transient errors (timeouts, malformed responses) with backoff before the
    caller's own try/except degrades the step."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return load_content(api.model_chat_completion(payload))
        except Exception as exc:  # noqa: BLE001 — retry transient errors, then re-raise
            last_exc = exc
            if attempt < attempts:
                log.warning(
                    "%s model call attempt %s/%s failed: %s%s",
                    _tag("REACT_FAIL"), attempt, attempts, exc, _line_end(),
                )
                time.sleep(min(2.0 * attempt, 5.0))
    raise last_exc if last_exc else RuntimeError("model call failed")
