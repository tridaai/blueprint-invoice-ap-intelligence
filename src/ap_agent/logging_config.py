"""Structured JSON logging with request IDs.

Production services log machine-readable lines. ``configure_logging``
installs a JSON formatter on the root logger (one object per line:
timestamp, level, logger, message, plus ``request_id`` when the
emitting code runs inside a request). The API middleware mints a
request id per request (honouring an inbound ``X-Request-ID``),
threads it through a context variable, and returns it on the
response — so a run can be followed from HTTP request to pipeline
events by one id.

Set ``LOG_FORMAT=text`` for human-readable local logs; JSON is the
default (the production posture). Configuration is env-only, like
everything else in this service.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from .config import env_str

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    return request_id_var.get()


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Install the configured formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    if env_str("LOG_FORMAT", "json").strip().lower() == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(env_str("LOG_LEVEL", "INFO").strip().upper() or "INFO")
