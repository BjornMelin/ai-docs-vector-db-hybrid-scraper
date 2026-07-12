# syntax=docker/dockerfile:1

# =========================================
# Stage 1: Build Environment
# =========================================
FROM python:3.11 AS builder

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Set working directory
WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install UV - the modern Python package manager
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /uvx /usr/local/bin/

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies with UV
RUN uv python install 3.11
RUN uv venv /opt/venv --python 3.11
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install locked dependencies into the environment copied into the runtime image.
# Application source is copied later and imported through PYTHONPATH.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Install the project after dependencies so distribution metadata and runtime
# version reporting come from pyproject.toml.
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =========================================
# Stage 2: Runtime Environment
# =========================================
FROM mcr.microsoft.com/playwright/python:v1.57.0-noble AS runtime

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-traditional \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
# Copy the UV Python installation to ensure interpreter compatibility
COPY --from=builder /opt/uv/python /opt/uv/python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Use the image's non-root UID 1000 user so bind-mounted workspace data remains writable.
RUN chown -R ubuntu:ubuntu /app

# Switch to non-root user
USER ubuntu

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Verify dependency construction and FastAPI lifespan as the runtime user
RUN python - <<'PY'
import asyncio

from src.api import app_factory
from src.infrastructure import container as container_module
from playwright.async_api import async_playwright


async def skip_service_hook(*_args, **_kwargs):
    return None


container_module._initialize_service_graph = skip_service_hook
container_module._cleanup_service_graph = skip_service_hook
app_factory._initialize_services = skip_service_hook


async def smoke():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        await browser.close()

    app = app_factory.create_app()
    async with asyncio.timeout(10):
        async with app.router.lifespan_context(app):
            container = app_factory.get_app_container(app)
            embedding_manager = container.embedding_manager()
            assert embedding_manager.config is container.config()
        assert app.state.container is None


asyncio.run(smoke())
PY

# Run the FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
