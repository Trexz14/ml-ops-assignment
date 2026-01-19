#### previosly in README
## Logging

This project uses **[loguru](https://github.com/Delgan/loguru)** for comprehensive logging across all components.

### Log Files

Logs are automatically written to the `logs/` directory:

- **`logs/app.log`**: General application logs (all levels: DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Rotates at 10 MB file size
  - Keeps logs for 7 days
  - Old logs are compressed as `.zip` files

- **`logs/errors.log`**: Error and critical messages only
  - Rotates at 5 MB file size
  - Keeps logs for 14 days
  - Includes full tracebacks and diagnostic information

### Log Levels

Control the logging verbosity by setting the `LOG_LEVEL` environment variable:

```bash
# Show all logs including DEBUG messages
export LOG_LEVEL=DEBUG

# Show only INFO and above (default)
export LOG_LEVEL=INFO

# Show only warnings and errors
export LOG_LEVEL=WARNING
```

### What Gets Logged

**Data Processing (`data.py`)**:
- Dataset loading from Hugging Face
- Tokenization progress
- Data splitting (train/val/test)
- Dataset saving

**Model Training (`model.py`, `train.py`)**:
- Configuration loading
- Device selection (CUDA/MPS/CPU)
- Model initialization
- Training progress (epoch summaries, batch progress)
- Validation metrics
- Checkpoint saving
- W&B integration status

**Model Evaluation (`evaluate.py`)**:
- Model loading
- Evaluation progress
- Final metrics (loss, accuracy)

**API Server (`api.py`)**:
- Application startup/shutdown
- Model and tokenizer loading
- Prediction requests
- Inference results

### Example Usage

All modules automatically use the centralized logging configuration. You'll see color-coded output in the console and detailed logs in files:

```bash
# Run training - logs go to console and logs/app.log
uv run python src/ml_ops_assignment/train.py

# Run evaluation - check logs/app.log for detailed progress
uv run python src/ml_ops_assignment/evaluate.py models/model_final.pt

# Start API server - logs requests to logs/app.log
uv run uvicorn ml_ops_assignment.api:app --host 0.0.0.0 --port 8000
```

### Docker Logging

When running in Docker containers, logs are still written to the `logs/` directory. Mount this directory to persist logs:

```bash
# Intel Mac users: Train with Docker and persist logs
docker run -v $(pwd)/logs:/app/logs ml-ops-train:latest

# Run API with log persistence
docker run -v $(pwd)/logs:/app/logs -p 8000:8000 ml-ops-api:latest
```

### Viewing Logs

```bash
# View recent logs
tail -f logs/app.log

# View only errors
tail -f logs/errors.log

# Search for specific information
grep "Epoch" logs/app.log

# View compressed old logs
unzip -p logs/app.log.2025-01-18.zip
```

---