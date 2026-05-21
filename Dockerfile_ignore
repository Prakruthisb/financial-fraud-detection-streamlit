# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install dependencies in an isolated layer so the final image stays lean.
# Using slim variant — full python image adds ~400MB we don't need.
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools needed to compile XGBoost / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer until requirements change.
# Changing app code won't re-trigger a pip install, saving minutes on rebuilds.
COPY requirements-prod.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-prod.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Start fresh — copy only installed packages and app code, not build tools.
# This keeps the final image ~40% smaller than a single-stage build.
FROM python:3.10-slim AS runtime

WORKDIR /app

# Non-root user — never run ML APIs as root in production
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application files
# COPY main.py           ./main.py
# COPY fraud_pipeline.py ./fraud_pipeline.py
# COPY fraud_pipeline.pkl ./fraud_pipeline.pkl
COPY . .

RUN chmod +x start.sh

ENV PATH="/usr/local/bin:$PATH"

# Switch to non-root user before starting the app
USER appuser

# Expose the port FastAPI will listen on
EXPOSE 10000

# Health check — Docker will mark container unhealthy if /health stops responding.
# --interval: check every 30s
# --timeout:  fail the check if no response within 10s
# --retries:  mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')"

# Start with uvicorn.
# --host 0.0.0.0   makes the API reachable from outside the container
# --workers 2      two processes for parallelism (tune to CPU count)
# --no-access-log  cleaner logs — remove this if you want per-request logs
CMD ["./start.sh"]