"""Shared MCP validation, auth, rate-limit and audit helpers."""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

DEFAULT_TOKEN = "helpdeskai-dev-token"
MAX_CALLS = 10
WINDOW_S = 60
CALL_HISTORY: dict[str, deque[float]] = defaultdict(deque)


def expected_token() -> str:
    """Return the shared MCP token for this POC."""
    return os.environ.get("HELPDESKAI_MCP_TOKEN", DEFAULT_TOKEN)


def configure_audit_logger(path: Path | None = None) -> logging.Logger:
    """Configure a JSON-lines audit logger."""
    audit_path = path or Path(os.environ.get("HELPDESKAI_MCP_AUDIT_PATH", "data/audit/mcp.jsonl"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("helpdeskai.mcp.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.FileHandler(str(audit_path), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


audit_logger = configure_audit_logger()


def reset_rate_limits() -> None:
    """Clear in-memory rate-limit state for tests."""
    CALL_HISTORY.clear()


def rate_limit_check(
    actor_id: str,
    *,
    max_calls: int = MAX_CALLS,
    window_s: int = WINDOW_S,
) -> tuple[bool, int]:
    now = time.time()
    history = CALL_HISTORY[actor_id]
    while history and history[0] < now - window_s:
        history.popleft()
    if len(history) >= max_calls:
        retry = int(window_s - (now - history[0]))
        return False, max(retry, 1)
    history.append(now)
    return True, 0


def require_token(token: str) -> None:
    if token != expected_token():
        raise PermissionError("invalid MCP token")


def audited_tool(
    tool_name: str,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Wrap a pure tool function with auth, rate-limit and JSON audit."""

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(
            *,
            actor_id: str = "agent_default",
            token: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            trace_id = str(uuid.uuid4())
            started_at = time.perf_counter()

            ok, retry = rate_limit_check(actor_id)
            if not ok:
                audit_logger.info(
                    json.dumps(
                        {
                            "event": "rate_limited",
                            "tool": tool_name,
                            "actor_id": actor_id,
                            "trace_id": trace_id,
                            "ts": time.time(),
                        }
                    )
                )
                return {"error": "rate_limited", "retry_after_s": retry}

            log = {
                "event": "tool_call",
                "tool": tool_name,
                "actor_id": actor_id,
                "trace_id": trace_id,
                "args": kwargs,
                "ts": time.time(),
            }
            try:
                require_token(token)
                result = fn(actor_id=actor_id, **kwargs)
                log.update(
                    {
                        "result": "success",
                        "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    }
                )
                audit_logger.info(json.dumps(log))
                return result
            except ValidationError as exc:
                log.update(
                    {
                        "result": "validation_error",
                        "errors": exc.errors(),
                        "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    }
                )
                audit_logger.info(json.dumps(log, default=str))
                return {"error": "validation_error", "details": exc.errors()}
            except PermissionError as exc:
                log.update(
                    {
                        "result": "auth_error",
                        "error_msg": str(exc),
                        "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    }
                )
                audit_logger.info(json.dumps(log))
                return {"error": "unauthorized", "details": str(exc)}
            except Exception as exc:
                log.update(
                    {
                        "result": "error",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                        "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    }
                )
                audit_logger.info(json.dumps(log))
                return {"error": "internal_error", "details": str(exc)}

        return wrapper

    return decorator
