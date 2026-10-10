# Invoice / AP Document Intelligence Agent — production-shaped image.
#
# Multi-stage: the build stage resolves the locked dependency set with
# uv; the runtime stage is a slim Python image running as a non-root
# user, with no build tooling. The container is STATELESS: approvals
# and graph checkpoints live in PostgreSQL, policy vectors in pgvector
# (same database), document blobs in S3-compatible object storage.
#
# NOTE: this image has not been built in the blueprint's dev sandbox
# (no Docker daemon there) — `docker compose build` is the first-run
# verification step, exactly as the README's QA checklist states.

FROM python:3.12-slim AS build

# uv for the locked install (single static binary from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md constraints.txt uv.lock ./
COPY src ./src
# Production image: provider SDKs + Postgres driver + S3 client.
ARG UV_EXTRA_ARGS="--extra llm --extra postgres --extra s3"
RUN uv sync --frozen --no-dev $UV_EXTRA_ARGS

FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY data ./data

ENV PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "ap_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
