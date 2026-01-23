# Text Quality Classification - MLOps Project

A machine learning system for classifying English text readability levels (Elementary, Intermediate, Advanced) using a fine-tuned BERT model. This project implements a complete MLOps pipeline including training, evaluation, API serving, and cloud deployment.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **⚠️ Intel Mac Users:** This project uses PyTorch 2.6.0, which doesn't have wheels for Intel Macs with `uv`. 
> Please see [INTEL_MAC_GUIDE.md](INTEL_MAC_GUIDE.md) for Docker-based setup instructions.

## Made By

**Group 63** - DTU Course 02476 Machine Learning Operations

| Student Number |
|----------------|
| s234869 (Alexander Hougaard) (GitHub: Alexander-bit-boop)|
| s245176 (William Hyldig) (GitHub: Williamhyldig)         |
| s244742 (Valdemar Stamm) (GitHub: HrStamm)               |
| s245362 (Frederik Jønsson) (GitHub: Trex14)              |
| s246089 (Gustav Christensen) (GitHub: DonConarch)        |

## Overview

This project was developed as part of the **02476 Machine Learning Operations** course at DTU. The goal is to build a production-ready ML pipeline that predicts text readability levels on a 0-2 scale:

- **0**: Elementary (simple vocabulary and sentence structure)
- **1**: Intermediate (moderate complexity)
- **2**: Advanced (complex vocabulary and sentence structure)

### Key Features

- 🤖 **BERT-based model** using the lightweight `prajjwal1/bert-mini` (11M parameters)
- 📊 **Experiment tracking** with Weights & Biases
- 🐳 **Containerized** training and API serving with Docker
- ☁️ **Cloud deployment** on Google Cloud Platform (Cloud Run)
- 🔄 **CI/CD pipelines** with GitHub Actions
- 📦 **Data versioning** with DVC and Google Cloud Storage
- 🧪 **Comprehensive testing** with pytest and code coverage
- 📈 **API monitoring** with Prometheus metrics

## Quick Start

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md).

```bash
# Clone the repository
git clone <repo-url>
cd ml-ops-assignment

# Install dependencies (requires uv)
uv sync

# Authenticate with Google Cloud (for data/model access)
gcloud auth application-default login

# Pull data and models from DVC
uv run dvc pull

# Evaluate the trained model
uv run invoke evaluate --checkpoint models/model_final.pt

# Start the API server
uv run invoke serve-api
```

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Pipeline                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   HuggingFace │────▶│  Tokenizer   │────▶│  Processed   │                 │
│  │    Dataset    │     │  (BERT)      │     │   Dataset    │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                    │                         │
│                                                    ▼                         │
│                              ┌──────────────────────────────────┐           │
│                              │         DVC + GCS Storage        │           │
│                              └──────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Training Pipeline                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Config     │────▶│   Training   │────▶│    Model     │                 │
│  │   (YAML)     │     │   Script     │     │  Checkpoints │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                              │                     │                         │
│                              ▼                     ▼                         │
│                       ┌────────────┐        ┌────────────┐                  │
│                       │   W&B      │        │    DVC     │                  │
│                       │  Logging   │        │   Storage  │                  │
│                       └────────────┘        └────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Serving Pipeline                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   FastAPI    │────▶│   Docker     │────▶│  Cloud Run   │                 │
│  │   Server     │     │   Container  │     │  Deployment  │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │  Prometheus  │                                                           │
│  │   Metrics    │                                                           │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             CI/CD Pipeline                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   GitHub     │────▶│   GitHub     │────▶│   Cloud      │                 │
│  │   Push       │     │   Actions    │     │   Build      │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                              │                     │                         │
│                              ▼                     ▼                         │
│                       ┌────────────┐        ┌────────────┐                  │
│                       │   Tests    │        │  Artifact  │                  │
│                       │  + Lint    │        │  Registry  │                  │
│                       └────────────┘        └────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```txt
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       ├── tests.yaml        # Run tests on push/PR
│       ├── linting.yaml      # Code linting with ruff
│       └── docker-build.yaml # Build and push Docker images
├── configs/                  # Experiment configuration files
│   └── experiments/
│       └── default.yaml      # Default training config
├── data/                     # Data directory (DVC tracked)
│   └── processed/            # Tokenized dataset
├── dockerfiles/              # Docker configurations
│   ├── api.dockerfile        # API server image
│   ├── train.dockerfile      # Training image
│   └── evaluate.dockerfile   # Evaluation image
├── docs/                     # MkDocs documentation
│   └── source/
├── models/                   # Trained model checkpoints (DVC tracked)
├── reports/                  # Exam report and figures
├── src/ml_ops_assignment/    # Source code
│   ├── api.py                # FastAPI application
│   ├── data.py               # Data loading and preprocessing
│   ├── evaluate.py           # Model evaluation script
│   ├── model.py              # Model definition and training logic
│   ├── train.py              # Training entry point
│   └── visualize.py          # Visualization utilities
├── tests/                    # Unit tests
│   ├── test_api.py           # API tests
│   ├── test_data.py          # Data pipeline tests
│   └── test_model.py         # Model tests
├── cloudbuild.yaml           # Google Cloud Build configuration
├── pyproject.toml            # Project dependencies (uv)
├── tasks.py                  # Invoke tasks for common commands
└── README.md                 # This file
```

## Usage

### Training

```bash
# Preprocess data (if starting from scratch)
uv run invoke preprocess-data

# Train the model
uv run invoke train

# Train with Docker
uv run invoke docker-train
```

### Evaluation

```bash
# Evaluate on test set
uv run invoke evaluate --checkpoint models/model_final.pt

# Evaluate on validation set
uv run invoke evaluate --checkpoint models/model_final.pt --split validation
```

### API

```bash
# Start the API server locally
uv run invoke serve-api

# Make a prediction
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=ml_ops_assignment --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Technology Stack

| Category | Tools |
|----------|-------|
| **ML Framework** | PyTorch 2.6.0, Transformers |
| **Model** | BERT-mini (prajjwal1/bert-mini) |
| **Data** | HuggingFace Datasets, DVC |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Monitoring** | Prometheus, Weights & Biases |
| **Infrastructure** | Docker, Google Cloud Run, Cloud Build |
| **CI/CD** | GitHub Actions |
| **Code Quality** | Ruff, Mypy, Pre-commit |
| **Testing** | Pytest, Coverage |
| **Package Management** | uv |

## Dataset

The project uses the [OneStop English](https://huggingface.co/datasets/SetFit/onestop_english) dataset from HuggingFace, which contains English texts at three readability levels. The dataset is automatically split into:

- **Training**: 80% of the data
- **Validation**: 10% of the data
- **Test**: 10% of the data

## Acknowledgments

- Course instructors and TAs at DTU
- [MLOps Template](https://github.com/SkafteNicki/mlops_template) by Nicki Skafte
- [HuggingFace](https://huggingface.co/) for the dataset and transformers library
- [Astral](https://astral.sh/) for uv package manager

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
