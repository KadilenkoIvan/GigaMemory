# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Which PyTorch variant: "cuda" (default) or "cpu"
# Override: --build-arg TORCH_EXTRA=cpu
ARG TORCH_EXTRA=cuda

WORKDIR /build

RUN pip install --no-cache-dir uv

# Copy everything needed to resolve + build the package
COPY pyproject.toml README.md ./
COPY RAGU/ ./RAGU/
COPY DST_memory/ ./DST_memory/

# Install project deps for the chosen torch variant into the system Python
RUN uv pip install --system --no-cache ".[${TORCH_EXTRA}]"

# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source (editable RAGU install path must stay at /app/RAGU)
COPY --chown=appuser:appuser DST_memory/ ./DST_memory/
COPY --chown=appuser:appuser RAGU/ ./RAGU/

USER appuser

ENV PYTHONPATH=/app/DST_memory

ENTRYPOINT ["python", "DST_memory/run.py"]