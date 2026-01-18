# Quick Start

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
