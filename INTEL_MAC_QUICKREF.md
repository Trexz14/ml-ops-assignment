# Intel Mac Quick Reference

Quick commands for Intel Mac users running this project with Docker.

## Prerequisites
- Docker Desktop installed and running
- Data downloaded: Run `dvc pull` (may need to install dvc with pip if uv fails)
- Google Cloud authenticated: `gcloud auth application-default login`

## One-Time Setup

```bash
# Build Docker images (takes 2-3 minutes first time)
docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile
docker build -t train:latest . -f dockerfiles/train.dockerfile
docker build -t api:latest . -f dockerfiles/api.dockerfile

# Or build all at once (if you can use invoke without torch)
uv run invoke docker-build
```

## Common Commands

### Evaluate Model

```bash
# Test set
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt test

# Validation set (if exists)
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt validation

# Different checkpoint
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_epoch_10.pt test
```

### Train Model

```bash
# Preprocess data first (if needed)
docker run --rm \
  --entrypoint uv \
  -v $(pwd)/data:/app/data \
  train:latest run python src/ml_ops_assignment/data.py process

# Train model
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  train:latest
```

### Run API

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  api:latest

# Access at: http://localhost:8000/docs
```

## Troubleshooting

### Docker not running
**Error:** `Cannot connect to the Docker daemon`
**Fix:** Start Docker Desktop application

### Image not found
**Error:** `Unable to find image 'evaluate:latest' locally`
**Fix:** Build the image first:
```bash
docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile
```

### Models not found
**Error:** `FileNotFoundError: models/model_final.pt`
**Fix:** Download models:
```bash
# Install dvc if needed
pip install dvc dvc-gs

# Download models and data
dvc pull
```

### Permission denied
**Fix:** On Linux, you may need to add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

## Advanced

### Interactive Shell in Container
```bash
docker run --rm -it \
  -v $(pwd):/app \
  --entrypoint /bin/bash \
  evaluate:latest
```

### Run Tests
```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  evaluate:latest \
  uv run pytest tests/ -v
```

### Clean Up Docker
```bash
# Remove unused images and containers
docker system prune -a

# Remove specific image
docker rmi evaluate:latest

# List all images
docker images
```

## Notes

- Volume mounts (`-v`) map your local files into the container
- `--rm` automatically removes container after it exits
- `-p 8000:8000` maps container port 8000 to your machine's port 8000
- All data/models stay on your machine (not in the container)

## Full Documentation

For complete details, see [INTEL_MAC_GUIDE.md](INTEL_MAC_GUIDE.md)
