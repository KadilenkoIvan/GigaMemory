# ─── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Which PyTorch variant to install: "cuda" (default, needs nvidia-container-toolkit)
# or "cpu" (CI, CPU-only).  Override: --build-arg TORCH_EXTRA=cpu
ARG TORCH_EXTRA=cuda

WORKDIR /build

RUN pip install --no-cache-dir uv

# Copy everything uv needs to resolve the dependency graph
COPY pyproject.toml README.md ./
COPY RAGU/ ./RAGU/
COPY DST_memory/ ./DST_memory/

# Always include the "api" extra (fastapi, uvicorn, matplotlib, networkx).
# TORCH_EXTRA selects the PyTorch wheel set (cuda vs cpu).
RUN uv pip install --system --no-cache ".[${TORCH_EXTRA},api]"

# ─── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# curl is needed for the HEALTHCHECK and for ad-hoc debugging inside the container
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source.
# RAGU must live at /app/RAGU so that api.py's _ensure_ragu_on_path() finds it.
COPY --chown=appuser:appuser DST_memory/ ./DST_memory/
COPY --chown=appuser:appuser RAGU/ ./RAGU/

USER appuser

ENV PYTHONPATH=/app/DST_memory

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Run uvicorn directly — no uv indirection needed since everything is installed
# into the system Python.  The config path is controlled via GIGAMEMORY_CONFIG.
ENTRYPOINT ["uvicorn", "api:app", "--app-dir", "DST_memory", "--host", "0.0.0.0", "--port", "8000"]
