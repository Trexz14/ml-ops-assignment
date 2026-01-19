# Quick Start

> **Note for Intel Mac Users:** If you're using an Intel Mac and encounter PyTorch installation issues with `uv`, see [INTEL_MAC_GUIDE.md](INTEL_MAC_GUIDE.md) for Docker-based instructions.

## Setup
```bash
# Clone repo
git clone <repo-url>
cd ml-ops-assignment

# Install dependencies
uv sync
```

## Authentication (First time only)
```bash
# Authenticate with Google Cloud
gcloud auth application-default login
```

## Get Data & Models
```bash
# Download processed data from GCS
uv run dvc pull
```

## Evaluate Model
```bash
# Test set (default)
uv run invoke evaluate --checkpoint models/model_final.pt

# Validation set
uv run invoke evaluate --checkpoint models/model_final.pt --split validation

# Specific checkpoint
uv run invoke evaluate --checkpoint models/model_epoch_10.pt
```

**Intel Mac Users (Docker):**
```bash
# Build Docker image (first time only)
uv run invoke docker-build

# Test set (default)
uv run invoke docker-evaluate --checkpoint models/model_final.pt

# Validation set
uv run invoke docker-evaluate --checkpoint models/model_final.pt --split validation
```

## Train New Model
```bash
# Preprocess data
uv run invoke preprocess-data

# Train model
uv run invoke train
```

## Inference API
```bash
# Start the API server
uv run invoke serve-api

# The API will be available at http://127.0.0.1:8000
# Documentation (Swagger UI) at http://127.0.0.1:8000/docs
```

## Run Tests
```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=ml_ops_assignment --cov-report=term-missing
```

## Format & Lint
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Upload Models to GCS
```bash
# Push to DVC remote
gsutil -m rsync -r .dvc/cache gs://mlops-dtu-data/dvc-cache/
```

## Docker (Alternative / Intel Mac)
```bash
# Build all Docker images
uv run invoke docker-build

# Or build individually
docker build -t train:latest . -f dockerfiles/train.dockerfile
docker build -t api:latest . -f dockerfiles/api.dockerfile
docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile

# Run evaluation via Docker
uv run invoke docker-evaluate --checkpoint models/model_final.pt

# See INTEL_MAC_GUIDE.md for complete Docker workflow
```
