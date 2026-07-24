FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

CMD ["uv", "run", "--no-sync", "fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]