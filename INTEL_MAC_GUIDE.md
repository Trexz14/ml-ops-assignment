# Intel Mac Setup Guide

**For users on macOS with Intel processors (x86_64)**

PyTorch doesn't provide wheels for Intel Macs. This guide shows you how to run everything using Docker.

## Prerequisites

Install these before starting:

1. **Docker Desktop** - https://www.docker.com/products/docker-desktop
2. **DVC** (for downloading data) - `pip install dvc dvc-gs`
3. **Google Cloud CLI** (optional, for data access) - For `gcloud` commands

## Setup (One Time)

### 1. Install Dependencies

```bash
uv sync
```

This installs all dependencies except PyTorch (which isn't available for Intel Mac).

### 2. Download Data and Models

```bash
dvc pull
```

### 3. Build Docker Images (Optional - Auto-Builds When Needed)

**You can skip this step!** The evaluate and train commands will automatically build the images if they don't exist.

**But if you want to build everything upfront:**
```bash
uv run invoke docker-build
```

**What this does:**
- Builds 3 Docker images that contain PyTorch and all dependencies
- Takes 10-30+ minutes the first time (downloads several GB of CUDA packages)
- Only needs to be done **once** - Docker caches the images for reuse
- Subsequent rebuilds are fast (only if you change code or dependencies)

The images created:
- `evaluate:latest` - For evaluating models
- `train:latest` - For training models  
- `api:latest` - For running the API

## Daily Usage

### Evaluate Models

**Easiest way:**
```bash
uv run invoke docker-evaluate --checkpoint models/model_final.pt
```

**Or with direct Docker:**
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt test
```

**Different checkpoints:**
```bash
# Use any checkpoint file
uv run invoke docker-evaluate --checkpoint models/model_epoch_5.pt
```

### Train Models

**Easiest way:**
```bash
uv run invoke docker-train
```

This trains a model using the default configuration (`configs/exp1.yaml`). New models are saved to `models/`.

**Train with different parameters:**
1. Edit `configs/exp1.yaml` or create a new config file
2. Run the training command:
```bash
uv run invoke docker-train
```

**Or with direct Docker (advanced):**
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  train:latest
```
or (recommended)
``` 
docker run --rm \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/configs":/app/configs \
  train:latest --config-path configs/experiments/exp2.yaml
```
Where "configs/experiments/exp2.yaml" is the path to the config file you want to train using.

**After training**, evaluate your new model:
```bash
uv run invoke docker-evaluate --checkpoint models/model_epoch_10.pt
```

### Run API Server

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  api:latest
```

Then open http://localhost:8000/docs in your browser.

### Format and Lint Code

These work directly (no Docker needed):

```bash
# Format code
uv run ruff format .

# Check for issues
uv run ruff check . --fix

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Common Issues

### Docker not running
**Error:** `Cannot connect to the Docker daemon`

**Fix:** Open Docker Desktop and wait for it to start.

---

### Image not found
**Error:** `Unable to find image 'evaluate:latest'`

**Fix:** Build the images first:
```bash
uv run invoke docker-build
```

---

### Models not found
**Error:** `FileNotFoundError: models/model_final.pt`

**Fix:** Download the models:
```bash
dvc pull
```

---

### Out of disk space
**Error:** `no space left on device`

**Fix:** Clean up Docker:
```bash
docker system prune -a
```

## Understanding the Commands

**Volume mounts** (`-v` flags) let Docker access your files:
- `-v $(pwd)/models:/app/models` - Maps your `models/` folder into the container
- `-v $(pwd)/data:/app/data` - Maps your `data/` folder into the container

**Other flags:**
- `--rm` - Removes the container after it exits (keeps things clean)
- `-p 8000:8000` - Maps port 8000 from container to your Mac

## What Works

| Command | Works? | Notes |
|---------|--------|-------|
| `uv sync` | ✅ Yes | Installs everything except PyTorch |
| `uv run invoke docker-build` | ✅ Yes | Builds Docker images (auto-happens if needed) |
| `uv run invoke docker-evaluate` | ✅ Yes | Easiest way to evaluate models |
| `uv run invoke docker-train` | ✅ Yes | Easiest way to train models |
| `uv run ruff format .` | ✅ Yes | Code formatting works directly |
| `uv run invoke evaluate` | ❌ No | Use `docker-evaluate` instead |
| `uv run invoke train` | ❌ No | Use `docker-train` instead |

## Tips

- **Docker images build automatically** - First time you run `docker-evaluate` or `docker-train`, images are built automatically if missing.
- **Images are reused** - After the first build, everything runs fast. No need to rebuild unless dependencies change.
- **Models stay on your Mac** - Docker just reads/writes them via volume mounts, doesn't copy them.
- **Use invoke commands** - `uv run invoke docker-train` and `docker-evaluate` are simpler than typing full Docker commands.
- **Experiment freely** - Edit configs, train, evaluate. Your data and models persist on your Mac between runs.

## Quick Reference

For a cheat sheet of common commands, see [INTEL_MAC_QUICKREF.md](INTEL_MAC_QUICKREF.md).
