# Intel Mac Quick Reference

**Fast command reference for Intel Mac users**

See [INTEL_MAC_GUIDE.md](INTEL_MAC_GUIDE.md) for detailed explanations.

## Prerequisites

- ✅ Docker Desktop installed and running
- ✅ Data downloaded: `pip install dvc dvc-gs && dvc pull`
- ✅ Dependencies installed: `uv sync`

## One-Time Setup

```bash
# Install dependencies (PyTorch skipped on Intel Mac)
uv sync

# Download data and models
dvc pull

# Build Docker images (optional - auto-builds when needed)
uv run invoke docker-build
```

⏱️ First build: 10-30 minutes. Subsequent builds: fast (cached).

## Common Commands

### Evaluate Models

```bash
# Test set (easiest way)
uv run invoke docker-evaluate --checkpoint models/model_final.pt

# Validation set
uv run invoke docker-evaluate --checkpoint models/model_final.pt --split validation

# Different checkpoint
uv run invoke docker-evaluate --checkpoint models/model_epoch_5.pt
```

**Or with direct Docker:**
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt test
```

### Train Models

```bash
# Easiest way (uses configs/exp1.yaml)
uv run invoke docker-train

# After training, evaluate the new model
uv run invoke docker-evaluate --checkpoint models/model_epoch_10.pt
```

**Or with direct Docker:**
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  train:latest
```

**To train with different parameters:**
1. Edit `configs/exp1.yaml` (or create new config)
2. Run `uv run invoke docker-train`

### Run API Server

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  api:latest
```

Access at: **http://localhost:8000/docs**

### Format & Lint Code

```bash
# These work directly (no Docker needed)
uv run ruff format .
uv run ruff check . --fix
uv run pre-commit run --all-files
```

## Troubleshooting

### Docker not running
**Error:** `Cannot connect to the Docker daemon`  
**Fix:** Start Docker Desktop application

---

### Image not found
**Error:** `Unable to find image 'evaluate:latest'`  
**Fix:** Images build automatically, but you can manually build:
```bash
uv run invoke docker-build
```

---

### Models not found
**Error:** `FileNotFoundError: models/model_final.pt`  
**Fix:** Download models:
```bash
pip install dvc dvc-gs
dvc pull
```

---

### Clean up Docker
```bash
# Remove unused images/containers
docker system prune -a

# Remove specific image
docker rmi evaluate:latest
```

## What Works ✅

| Command | Works? | Notes |
|---------|--------|-------|
| `uv sync` | ✅ Yes | Installs everything except PyTorch |
| `uv run invoke docker-evaluate` | ✅ Yes | Auto-builds if needed |
| `uv run invoke docker-train` | ✅ Yes | Auto-builds if needed |
| `uv run ruff format .` | ✅ Yes | No Docker needed |
| `uv run invoke evaluate` | ❌ No | Use `docker-evaluate` instead |
| `uv run invoke train` | ❌ No | Use `docker-train` instead |

## Advanced Commands

### Interactive shell in container
```bash
docker run --rm -it \
  -v $(pwd):/app \
  --entrypoint /bin/bash \
  evaluate:latest
```

### Run tests
```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  evaluate:latest \
  uv run pytest tests/ -v
```

### List available tasks
```bash
uv run invoke --list
```

## Quick Workflow

**For new Intel Mac users:**
1. `uv sync` - Install dependencies
2. `dvc pull` - Download data/models
3. `uv run invoke docker-evaluate --checkpoint models/model_final.pt` - Run evaluation (auto-builds image)

**To experiment with training:**
1. Edit `configs/exp1.yaml` with your parameters
2. `uv run invoke docker-train` - Train model
3. `uv run invoke docker-evaluate --checkpoint models/model_epoch_10.pt` - Evaluate results

## Notes

- **Auto-build:** First time you run `docker-train` or `docker-evaluate`, images build automatically
- **Volume mounts** (`-v`) let Docker access your local files
- **Models persist:** All models/data stay on your Mac between runs
- **Fast iteration:** After first build, everything runs quickly

**See [INTEL_MAC_GUIDE.md](INTEL_MAC_GUIDE.md) for complete documentation.**
