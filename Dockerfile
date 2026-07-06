FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY helpdeskai ./helpdeskai
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "scripts/demo_streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
