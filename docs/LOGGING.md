# Logging Documentation

This document provides detailed information about the logging system implemented in this project.

## Overview

The project uses **[loguru](https://github.com/Delgan/loguru)** for structured, thread-safe logging across all components. All logging is centrally configured in `src/ml_ops_assignment/logging_config.py`.

## Architecture

### Centralized Configuration

All modules import the logger from `logging_config.py`:

```python
from ml_ops_assignment.logging_config import get_logger

logger = get_logger(__name__)
```

This ensures consistent formatting, log levels, and output destinations across the entire application.

### Log Handlers

The logging system uses multiple handlers:

1. **Console Handler (stderr)**
   - Colored output for better readability
   - Respects `LOG_LEVEL` environment variable
   - Format: `<time> | <level> | <module>:<function>:<line> - <message>`

2. **File Handler (app.log)**
   - All log levels (DEBUG and above)
   - 10 MB rotation with 7-day retention
   - Automatic compression to `.zip` files

3. **Error File Handler (errors.log)**
   - ERROR and CRITICAL only
   - 5 MB rotation with 14-day retention
   - Includes full tracebacks and diagnostic info

## Configuration

### Environment Variables

- `LOG_LEVEL`: Controls console output verbosity (default: `INFO`)
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Example:
```bash
# In .env file or shell
export LOG_LEVEL=DEBUG
```

### Customizing Log Location

To change the log directory, modify `logging_config.py`:

```python
LOG_DIR = Path("custom/log/path")
```

## Usage Examples

### Basic Logging

```python
from ml_ops_assignment.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Warning message for potentially harmful situations")
logger.error("Error message for serious problems")
logger.critical("Critical message for very serious errors")
logger.success("Success message (loguru-specific)")
```

### Logging with Context

```python
logger.info(f"Processing {batch_size} samples in batch {batch_idx}")
logger.info(f"Model achieved {accuracy:.2f}% accuracy on validation set")
```

### Exception Logging

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

Loguru automatically captures exception tracebacks when logging from an exception handler.

## Log Levels by Module

### data.py
- **INFO**: Dataset loading, tokenization, splitting progress
- **SUCCESS**: Dataset saved successfully
- **ERROR**: Dataset loading failures

### model.py
- **INFO**: Model initialization, device selection, training/evaluation progress
- **DEBUG**: Batch-level progress, checkpoint details
- **SUCCESS**: Model loaded, training completed
- **ERROR**: Configuration loading errors, model loading failures

### train.py
- **INFO**: Training script start, W&B initialization, configuration
- **SUCCESS**: Training completion, W&B run finished
- **WARNING**: Missing W&B credentials

### evaluate.py
- **INFO**: Evaluation start, model loading, final metrics
- **SUCCESS**: Evaluation completed

### api.py
- **INFO**: Application startup/shutdown, model loading, prediction requests
- **DEBUG**: Health checks, tokenization details, inference steps
- **WARNING**: Mock mode (model not loaded)
- **ERROR**: Model/tokenizer loading failures

## Docker Integration

Logs work seamlessly in Docker containers. To persist logs from Docker:

```bash
# Training
docker run -v $(pwd)/logs:/app/logs ml-ops-train:latest

# API
docker run -v $(pwd)/logs:/app/logs -p 8000:8000 ml-ops-api:latest

# Evaluation
docker run -v $(pwd)/logs:/app/logs ml-ops-evaluate:latest
```

The `logs/` directory is automatically created if it doesn't exist.

## Log Maintenance

### Viewing Logs

```bash
# Tail live logs
tail -f logs/app.log

# View errors only
tail -f logs/errors.log

# Search for specific events
grep "Training" logs/app.log
grep "ERROR" logs/app.log

# View specific time range
grep "2025-01-19 14:" logs/app.log
```

### Cleaning Old Logs

Logs are automatically rotated and compressed. Manual cleanup:

```bash
# Remove compressed logs older than 30 days
find logs/ -name "*.zip" -mtime +30 -delete

# Remove all logs (fresh start)
rm -rf logs/*.log logs/*.zip
```

### Log Size Management

If logs grow too large, adjust rotation settings in `logging_config.py`:

```python
logger.add(
    LOG_DIR / "app.log",
    rotation="5 MB",    # Rotate at 5 MB instead of 10 MB
    retention="3 days", # Keep for 3 days instead of 7
)
```

## Best Practices

1. **Use appropriate log levels**:
   - DEBUG: Detailed info for debugging (not in production)
   - INFO: Normal operations, milestones
   - WARNING: Unexpected but handled situations
   - ERROR: Errors that prevent specific operations
   - CRITICAL: System-level failures

2. **Include context**:
   ```python
   # Good
   logger.info(f"Processing batch {batch_idx}/{total_batches}")
   
   # Less useful
   logger.info("Processing batch")
   ```

3. **Log before expensive operations**:
   ```python
   logger.info("Starting model training...")
   model.train()
   logger.success("Training completed")
   ```

4. **Don't log sensitive data**:
   ```python
   # Bad
   logger.info(f"User password: {password}")
   
   # Good
   logger.info(f"User {username} authenticated")
   ```

## Troubleshooting

### Logs not appearing

1. Check `LOG_LEVEL` environment variable
2. Verify `logs/` directory exists and is writable
3. Check if another process has locked the log file

### Docker logs not persisting

1. Ensure volume mount: `-v $(pwd)/logs:/app/logs`
2. Check directory permissions
3. Verify `LOG_DIR` in `logging_config.py` matches Docker path

### Too many log files

1. Adjust `retention` parameter in `logging_config.py`
2. Implement log cleanup in deployment scripts
3. Use log aggregation services (e.g., CloudWatch, Elasticsearch)

## Performance Considerations

- **Async logging**: Enabled via `enqueue=True` (thread-safe, non-blocking)
- **Compression**: Old logs are automatically compressed to save space
- **Rotation**: Prevents unbounded log file growth
- **Lazy formatting**: Use f-strings for efficient string formatting

## Future Enhancements

Potential improvements to the logging system:

1. **Structured logging**: Add JSON format option for log aggregation
2. **Remote logging**: Send logs to centralized logging service
3. **Metrics integration**: Log metrics to Prometheus/Grafana
4. **Log filtering**: Per-module log level configuration
5. **Alerts**: Trigger alerts on ERROR/CRITICAL logs

## References

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Python Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [12-Factor App Logs](https://12factor.net/logs)
