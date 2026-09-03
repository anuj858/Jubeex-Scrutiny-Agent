FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . /app/

ENV PATH="/root/.local/bin:$PATH"

RUN uv sync --locked

RUN uv tool install llamactl

EXPOSE 4501

ENTRYPOINT ["llamactl", "serve", "--host", "0.0.0.0", "--port", "4501"]
