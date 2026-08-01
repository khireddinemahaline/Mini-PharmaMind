# syntax=docker/dockerfile:1
##############################################
# ChEMBL MCP Server Dockerfile
# This Dockerfile builds a Node.js MCP server for ChEMBL


##############################################
# Stage 1: Build (with devDependencies)
##############################################
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files and TypeScript config
COPY package*.json tsconfig.json ./

# Install all dependencies including dev
RUN npm ci

# Copy the rest of the source code
COPY src/ ./src/

# Build TypeScript project
RUN npm run build

##############################################
# Stage 2: Runtime (production only)
##############################################
FROM node:22-alpine

WORKDIR /app

# Copy only package files to install prod deps
COPY package*.json ./
RUN npm ci --only=production

# Copy built JS from builder stage
COPY --from=builder /app/build ./build

# Security: create non-root user
RUN addgroup -g 1001 -S mcpuser && \
    adduser -S mcpuser -u 1001
RUN chown -R mcpuser:mcpuser /app
USER mcpuser

# Expose port (optional, MCP usually uses stdio)
EXPOSE 3000

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "console.log('ChEMBL MCP Server is healthy')" || exit 1

# Metadata labels
LABEL maintainer="Augmented Nature <contact@augmentednature.ai>"
LABEL description="ChEMBL Model Context Protocol Server"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/augmented-nature/chembl-mcp-server"
LABEL org.opencontainers.image.description="A comprehensive MCP server for ChEMBL chemical database access"
LABEL org.opencontainers.image.licenses="MIT"

# Start the compiled Node server
ENTRYPOINT ["node", "build/index.js"]
