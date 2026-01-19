# Evaluate dockerfile for ml_ops_assignment
# Builds an image that can evaluate trained models
# Specifically useful for Intel Mac users where PyTorch wheels are unavailable for uv

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Install build essentials (needed for some Python packages)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for better Docker cache)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install dependencies (cached separately from code changes)
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Set entrypoint to run evaluation
# Arguments can be passed at runtime: docker run evaluate:latest <checkpoint> <split>
ENTRYPOINT ["uv", "run", "python", "-u", "src/ml_ops_assignment/evaluate.py"]
CMD ["models/model_final.pt", "test"]
