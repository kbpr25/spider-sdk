# ==============================================================================
# S.P.I.D.E.R. Arena - SWE-Bench Evaluation Container
# ==============================================================================
# 
# This Dockerfile creates a sealed environment for running S.P.I.D.E.R. against
# real-world bugs from open source projects (like Flask, Requests, Django).
#
# Usage:
#   docker build -t spider-arena -f spider/benchmarks/Dockerfile .
#   docker run -it spider-arena
#
# ==============================================================================

# Base: Python 3.12 slim for minimal footprint
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Version control
    git \
    # Build tools for native extensions
    build-essential \
    gcc \
    g++ \
    # Development headers
    python3-dev \
    libffi-dev \
    libssl-dev \
    # Process management
    procps \
    # Text editing (for debugging)
    vim-tiny \
    # Network tools
    curl \
    wget \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace directories
WORKDIR /app
RUN mkdir -p /app/workspace \
    /app/spider_sdk \
    /app/target_repo \
    /app/results

# Copy S.P.I.D.E.R. SDK into the container
COPY . /app/spider_sdk

# Install S.P.I.D.E.R. in editable mode
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e /app/spider_sdk

# Install common testing dependencies
RUN pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-timeout>=2.0.0 \
    coverage>=7.0.0

# Configure Git for the Agent
RUN git config --global user.email "spider@agent.ai" \
    && git config --global user.name "S.P.I.D.E.R. Agent" \
    && git config --global init.defaultBranch main \
    && git config --global advice.detachedHead false

# Create the benchmark runner script
RUN echo '#!/bin/bash\n\
    echo "🕷️ S.P.I.D.E.R. Arena Ready"\n\
    echo "Python: $(python --version)"\n\
    echo "Git: $(git --version)"\n\
    echo "Spider: $(spider status 2>/dev/null || echo "Available")"\n\
    ' > /app/arena_status.sh && chmod +x /app/arena_status.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spider; print(spider.__version__)" || exit 1

# Default working directory for benchmarks
WORKDIR /app/workspace

# Labels for container identification
LABEL maintainer="S.P.I.D.E.R. Team" \
    version="0.1.0" \
    description="Zero-cost SWE-Bench evaluation environment"

# Default entrypoint
CMD ["/bin/bash"]