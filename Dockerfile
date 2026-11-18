FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /usr/local/bin/uv
WORKDIR /opt/synthex
COPY . .
RUN uv sync --frozen --no-dev --extra manuscript

ENV PATH="/opt/synthex/.venv/bin:${PATH}"
ENTRYPOINT ["synthex"]
