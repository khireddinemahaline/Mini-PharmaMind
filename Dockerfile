# ============================================================================
# Mini-PharmaMind - Multi-Agent Pharmaceutical Research System
# ============================================================================
# Multi-stage Dockerfile optimized for production deployment
# 
# Build stages:
#   1. Builder: Compiles dependencies and creates virtual environment
#   2. Runtime: Minimal production image with security hardening
#
# Security features:
#   - Non-root user execution
#   - Minimal base image (slim-bookworm)
#   - Regular security updates
#   - No unnecessary build tools in runtime
#
# Usage:
#   docker build -t mini-pharmamind:latest .
#   docker run -p 8000:8000 mini-pharmamind:latest
# ============================================================================

##############################################
# Stage 1: Build stage
##############################################
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl ca-certificates libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files only
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies (no cache)
RUN /root/.local/bin/uv venv /app/.venv && \
    /root/.local/bin/uv pip install --python /app/.venv/bin/python -r pyproject.toml --no-cache

##############################################
# Stage 2: Runtime stage
##############################################
FROM python:3.10-slim-bookworm

# Metadata labels
LABEL maintainer="MHLAINE Khireddine <mhalaine.khireddine.chimie@gmail.com>" \
    description="Mini-PharmaMind: Lightweight multi-agent AI for pharmaceutical research (mini version)" \
    version="0.1.0-mini" \
    license="MIT" \
    url="https://github.com/khireddinemahaline/mini-pharmamind"

WORKDIR /app

# Install only essential runtime dependencies including Node.js for Prisma
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates dumb-init libpq5 libatomic1 \
    nodejs npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r pharma --gid=1000 && \
    useradd -r -g pharma --uid=1000 --home-dir=/app --shell=/bin/bash pharma

# Copy Python environment from builder stage
COPY --from=builder --chown=pharma:pharma /app/.venv /app/.venv

# Copy only necessary application files (exclude tests, docs, etc.)
COPY --chown=pharma:pharma agents ./agents
COPY --chown=pharma:pharma config ./config
COPY --chown=pharma:pharma orcastration ./orcastration
COPY --chown=pharma:pharma tools ./tools
COPY --chown=pharma:pharma utilities ./utilities
COPY --chown=pharma:pharma prisma ./prisma
COPY --chown=pharma:pharma public ./public
COPY --chown=pharma:pharma chainlit.md ./chainlit.md

# Setup required directories
RUN mkdir -p session_state generated_reports .chainlit && \
    chown -R pharma:pharma /app && \
    chmod -R 755 /app

# Clean up Python cache files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete && \
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + 2>/dev/null || true

# Security & env vars
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONFAULTHANDLER=1 \
    PYTHONMALLOC=malloc \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=staging

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["bash", "-c", "prisma db push && chainlit run orcastration/main_chainlit.py -w --host 0.0.0.0 --port 8000"]