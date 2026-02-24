# ╔══════════════════════════════════════════════════════════╗
# ║  Multi-stage Dockerfile — ~225MB production image       ║
# ╚══════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Production ────────────────────────────────────
FROM python:3.11-slim AS production

# Chrome for html2image (news card rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMIUM_FLAGS="--no-sandbox --headless --disable-gpu"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY cron/ ./cron/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create output directory
RUN mkdir -p output/images && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT:-8000}/healthz/')"

# Railway injects PORT; bind to [::] for IPv6 support
CMD ["sh", "-c", "python -m alembic upgrade head && gunicorn app.main:app --bind [::]:${PORT:-8000} --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120"]
