# API dockerfile for ml_ops_assignment
# Builds an image that serves the model via FastAPI

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

# Copy trained model (mount or include during build)
# Models can also be mounted at runtime: -v $(pwd)/models:/app/models
COPY models/ models/

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Expose port for the API
EXPOSE 8000

# Set entrypoint to run FastAPI server
ENTRYPOINT ["uv", "run", "uvicorn", "ml_ops_assignment.api:app", "--host", "0.0.0.0", "--port", "8000"]
