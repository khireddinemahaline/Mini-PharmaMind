# ============================================================================
# Mini-PharmaMind - Multi-Agent Pharmaceutical Research System
# ============================================================================
# Multi-stage Dockerfile optimized for production deployment
#
# MCP servers are copied from the LOCAL project directory:
#
#   mcp-servers/
#   ├── ChEMBL-MCP-Server/
#   └── OpenTargets-MCP-Server/
#
# No git clone is performed during the Docker build.
# ============================================================================

##############################################
# Stage 1: Build stage
##############################################
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy Python dependency files
COPY pyproject.toml uv.lock ./

# ------------------------------------------------------------
# Copy MCP servers from the LOCAL project directory.
#
# Expected local structure:
#
# mcp-servers/
# ├── ChEMBL-MCP-Server/
# └── OpenTargets-MCP-Server/
# ------------------------------------------------------------
COPY mcp-servers /app/mcp-servers

# Create Python virtual environment and install dependencies
RUN /root/.local/bin/uv venv /app/.venv && \
    /root/.local/bin/uv pip install \
    --python /app/.venv/bin/python \
    -r pyproject.toml \
    --no-cache

# ------------------------------------------------------------
# Build local MCP servers
# ------------------------------------------------------------

RUN cd /app/mcp-servers/ChEMBL-MCP-Server && \
    npm install && \
    npm run build

RUN cd /app/mcp-servers/OpenTargets-MCP-Server && \
    npm install && \
    npm run build


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

# ------------------------------------------------------------
# Runtime dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    dumb-init \
    libpq5 \
    libatomic1 \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create non-root user
# ------------------------------------------------------------
RUN groupadd -r pharma --gid=1000 && \
    useradd -r \
    -g pharma \
    --uid=1000 \
    --home-dir=/app \
    --shell=/bin/bash \
    pharma

# ------------------------------------------------------------
# Copy Python environment from builder
# ------------------------------------------------------------
COPY --from=builder \
    --chown=pharma:pharma \
    /app/.venv \
    /app/.venv

# ------------------------------------------------------------
# Copy locally provided and BUILT MCP servers
# ------------------------------------------------------------
COPY --from=builder \
    --chown=pharma:pharma \
    /app/mcp-servers \
    /app/mcp-servers

# ------------------------------------------------------------
# Copy application files
# ------------------------------------------------------------
COPY --chown=pharma:pharma agents ./agents
COPY --chown=pharma:pharma config ./config
COPY --chown=pharma:pharma orcastration ./orcastration
COPY --chown=pharma:pharma tools ./tools
COPY --chown=pharma:pharma utilities ./utilities
COPY --chown=pharma:pharma prisma ./prisma
COPY --chown=pharma:pharma public ./public
COPY --chown=pharma:pharma chainlit.md ./chainlit.md

# ------------------------------------------------------------
# Setup required directories
# ------------------------------------------------------------
RUN mkdir -p \
    session_state \
    generated_reports \
    .chainlit \
    && chown -R pharma:pharma /app \
    && chmod -R 755 /app

# ------------------------------------------------------------
# Clean Python cache files
# ------------------------------------------------------------
RUN find /app \
    -type f \
    -name "*.pyc" \
    -delete && \
    find /app \
    -type d \
    -name "__pycache__" \
    -delete && \
    find /app/.venv \
    -type d \
    -name "tests" \
    -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Network
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=40s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ------------------------------------------------------------
# Run as non-root
# ------------------------------------------------------------
USER pharma

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# ------------------------------------------------------------
# Start application
# ------------------------------------------------------------
CMD ["bash", "-c", "prisma db push && chainlit run orcastration/main_chainlit.py -w --host 0.0.0.0 --port 8000"]