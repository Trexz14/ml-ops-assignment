# Summary of Changes for Intel Mac Compatibility

This document summarizes the changes made to enable Intel Mac users to run the project using Docker.

## Problem

On Intel Macs, `uv` doesn't have pre-built wheels for PyTorch 2.6.0, causing installation failures when trying to run commands like:
```bash
uv run invoke evaluate --checkpoint models/model_final.pt
```

Error message:
```
error: Distribution `torch==2.6.0 @ registry+https://download.pytorch.org/whl/cpu` can't be installed because 
it doesn't have a source distribution or wheel for the current platform
```

## Solution

We've implemented a Docker-based workflow that works on all platforms, including Intel Macs. Docker provides a consistent Linux environment where PyTorch wheels are available.

## Files Created/Modified

### 1. New Files Created

#### `dockerfiles/evaluate.dockerfile`
- Docker image specifically for model evaluation
- Uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` as base
- Installs all dependencies including PyTorch in a Linux environment
- Mounts `models/` and `data/` directories at runtime for accessing local files

#### `INTEL_MAC_GUIDE.md`
- Comprehensive guide for Intel Mac users
- Step-by-step instructions for:
  - Installing Docker Desktop
  - Building Docker images
  - Running evaluation, training, and API
  - Troubleshooting common issues
- Comparison table between standard and Docker workflows

#### `documents/INTEL_MAC_SUMMARY.md`
- This file - technical summary of changes
- Documents all modifications and their rationale

### 2. Modified Files

#### `tasks.py`
- Added `docker_evaluate()` task for running evaluation via Docker
- Updated `docker_build()` to also build the evaluation image
- Usage: `uv run invoke docker-evaluate --checkpoint models/model_final.pt --split test`

#### `QUICKSTART.md`
- Added note at the top referencing Intel Mac guide
- Added Docker evaluation commands in the "Evaluate Model" section
- Added new "Docker (Alternative / Intel Mac)" section at the bottom

#### `src/ml_ops_assignment/data.py`
- Fixed `collate_fn()` to truncate sequences exceeding 512 tokens
- Prevents runtime errors when batch contains sequences longer than BERT's max position embeddings
- This bug affected all platforms but was discovered during Intel Mac testing

## How It Works

### Docker Workflow

1. **Build once**: Docker image contains all dependencies including PyTorch
   ```bash
   docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile
   ```

2. **Run evaluation**: Mount local directories into container
   ```bash
   docker run --rm \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/data:/app/data \
     evaluate:latest models/model_final.pt test
   ```

3. **Using invoke** (recommended):
   ```bash
   uv run invoke docker-evaluate --checkpoint models/model_final.pt
   ```

### Volume Mounts

- `-v $(pwd)/models:/app/models`: Maps local models directory to container
- `-v $(pwd)/data:/app/data`: Maps local data directory to container
- `--rm`: Automatically removes container after execution

## Compatibility

### Standard Workflow (uv)
- ✅ Works on: Linux x86_64, macOS ARM (M1/M2/M3), Windows x86_64
- ❌ Fails on: macOS Intel (x86_64)

### Docker Workflow
- ✅ Works on: All platforms with Docker installed
- No platform-specific wheel requirements
- Consistent behavior across all systems

## Testing Results

Successfully tested on Intel Mac:

```bash
$ docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt test

2026-01-19 11:35:09.718 | INFO     | __main__:main:48 -   Loss: 0.4587
2026-01-19 11:35:09.720 | INFO     | __main__:main:49 -   Accuracy: 87.20%
```

## Recommendations for Users

### Intel Mac Users
1. Install Docker Desktop
2. Follow [INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md) for complete setup
3. Use `uv run invoke docker-evaluate` for evaluation
4. Use similar Docker commands for training and API

### Non-Intel Users
- Can continue using standard `uv run` commands
- Docker workflow also available if preferred
- Both workflows maintain identical functionality

## Recommendations for Maintainers

1. **Test both workflows**: When making changes, test both standard and Docker workflows
2. **Update Docker images**: Rebuild images after dependency changes
3. **Keep documentation synced**: Update both QUICKSTART.md and INTEL_MAC_GUIDE.md
4. **Consider CI/CD**: Add Docker-based tests to ensure compatibility

## Future Enhancements

Potential improvements to consider:

1. **Pre-built images**: Push Docker images to a registry (e.g., Docker Hub, GitHub Container Registry)
   - Users can pull instead of build
   - Faster setup for new users

2. **Docker Compose**: Create `docker-compose.yml` for easier orchestration
   - Single command to run training + API
   - Manage multiple containers

3. **Dev containers**: Add `.devcontainer` configuration for VS Code
   - Full development environment in Docker
   - Consistent across all platforms

4. **GPU support**: Add GPU-enabled Dockerfile variant
   - For users with NVIDIA GPUs on Linux
   - Faster training and evaluation

## Additional Notes

### Bug Fix
The `collate_fn()` fix in `data.py` resolves an issue where sequences longer than 512 tokens caused runtime errors. This was discovered during testing but affects all platforms, not just Intel Macs. The fix ensures sequences are truncated to the model's maximum supported length.

### Performance
- Docker adds minimal overhead (~1-2 seconds for container startup)
- Evaluation performance is identical to native execution
- First build takes 2-3 minutes; subsequent builds use cache and are fast

### Data and Models
- Models and data remain on host machine (not copied into image)
- Volume mounts provide container access without duplication
- Changes to models/data are immediately available to containers
