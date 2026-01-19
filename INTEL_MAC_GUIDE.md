# Intel Mac User Guide

This guide is specifically for users running macOS on Intel chips who encounter issues with PyTorch and `uv` due to missing wheel files.

## The Problem

On Intel Macs, `uv` may not have pre-built wheels available for PyTorch, which causes errors when trying to run commands like:
```bash
uv run invoke evaluate --checkpoint models/model_final.pt
```

You might see errors such as:
- `No matching distribution found for torch`
- `Could not find a version that satisfies the requirement torch`

## The Solution: Docker

Docker provides a consistent, isolated environment that works regardless of your host architecture. We've created Docker-based workflows that allow you to run all project tasks seamlessly.

## Prerequisites

1. **Install Docker Desktop for Mac**
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Verify installation: `docker --version`

2. **Authenticate with Google Cloud** (for data access)
   ```bash
   gcloud auth application-default login
   ```

3. **Download Data and Models**
   ```bash
   # You can run this without uv
   dvc pull
   ```
   
   If `dvc` is not installed globally, you can use pip:
   ```bash
   pip install dvc dvc-gs
   dvc pull
   ```

## Quick Start for Intel Mac Users

### 1. Build Docker Images

First time only, build the Docker images:
```bash
# Build all images (train, api, and evaluate)
uv run invoke docker-build

# Or build just the evaluation image
docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile
```

This step takes a few minutes but only needs to be done once (or when dependencies change).

### 2. Evaluate a Model

Now you can evaluate models using Docker:

```bash
# Using invoke (recommended)
uv run invoke docker-evaluate --checkpoint models/model_final.pt

# Test set (default)
uv run invoke docker-evaluate --checkpoint models/model_final.pt --split test

# Validation set
uv run invoke docker-evaluate --checkpoint models/model_final.pt --split validation

# Different checkpoint
uv run invoke docker-evaluate --checkpoint models/model_epoch_10.pt
```

**Or run Docker directly:**
```bash
# Test set
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data evaluate:latest models/model_final.pt test

# Validation set
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data evaluate:latest models/model_final.pt validation
```

### 3. Train a Model

```bash
# Build training image if not already built
docker build -t train:latest . -f dockerfiles/train.dockerfile

# Run training
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs train:latest
```

### 4. Run the API

```bash
# Build API image if not already built
docker build -t api:latest . -f dockerfiles/api.dockerfile

# Run API server
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models api:latest

# Access at: http://localhost:8000/docs
```

## Understanding the Docker Commands

When running Docker commands, you'll see volume mounts like `-v $(pwd)/models:/app/models`. Here's what they mean:

- `-v $(pwd)/models:/app/models` - Mounts your local `models/` directory into the container so it can access your trained models
- `-v $(pwd)/data:/app/data` - Mounts your local `data/` directory for access to datasets
- `--rm` - Automatically removes the container after it finishes running
- `-p 8000:8000` - Maps port 8000 from the container to your host machine (for API access)

## Troubleshooting

### Docker not found
**Error:** `docker: command not found`

**Solution:** Install Docker Desktop from https://www.docker.com/products/docker-desktop and ensure it's running.

### Permission denied
**Error:** `permission denied while trying to connect to the Docker daemon socket`

**Solution:** Ensure Docker Desktop is running. On Mac, just start the Docker Desktop application.

### Cannot connect to Docker daemon
**Error:** `Cannot connect to the Docker daemon`

**Solution:** Start Docker Desktop application and wait for it to fully initialize.

### Models not found
**Error:** `FileNotFoundError: models/model_final.pt`

**Solution:** Ensure you've run `dvc pull` to download the models before running evaluation.

### Out of disk space
**Error:** `no space left on device`

**Solution:** Clean up unused Docker images and containers:
```bash
docker system prune -a
```

## Running Tests

Tests can still be run with pytest directly if you have Python installed via another method (e.g., Homebrew), or via Docker:

```bash
# Build a test image (you can use the train image)
docker run --rm -v $(pwd):/app -w /app train:latest uv run pytest tests/ -v
```

## Comparison: Standard vs Docker Workflow

| Task | Standard (uv) | Intel Mac (Docker) |
|------|---------------|-------------------|
| **Evaluate** | `uv run invoke evaluate --checkpoint models/model_final.pt` | `uv run invoke docker-evaluate --checkpoint models/model_final.pt` |
| **Train** | `uv run invoke train` | `docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data train:latest` |
| **API** | `uv run invoke serve-api` | `docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models api:latest` |
| **Tests** | `uv run pytest tests/` | `docker run --rm -v $(pwd):/app -w /app train:latest uv run pytest tests/` |

## Notes

- Docker containers are ephemeral - any changes inside the container are lost when it stops (unless saved to mounted volumes)
- Mounted volumes (`-v` flags) allow the container to access and modify files on your host machine
- The Docker images use the same `uv` and dependencies as the standard workflow, ensuring consistency
- Building images the first time takes several minutes, but subsequent runs are fast
- You can mix Docker and non-Docker commands - for example, use Docker for Python tasks but run `ruff` or `pre-commit` natively if you have them installed

## For Maintainers

If you're updating dependencies or code:

1. **Rebuild Docker images** after changing `pyproject.toml` or `uv.lock`:
   ```bash
   uv run invoke docker-build
   ```

2. **Test both workflows** to ensure compatibility:
   ```bash
   # Standard workflow (if you have compatible hardware)
   uv run invoke evaluate --checkpoint models/model_final.pt
   
   # Docker workflow (works on all platforms)
   uv run invoke docker-evaluate --checkpoint models/model_final.pt
   ```

3. **Keep documentation updated** in this file when adding new Docker-based commands.
