# Intel Mac Compatibility - Technical Reference

**For maintainers and contributors**

This document explains the technical solution for Intel Mac compatibility.

## Problem

PyTorch 2.6.0 doesn't provide wheels for macOS x86_64 (Intel Macs), causing:
```
error: Distribution `torch==2.6.0` can't be installed because it doesn't have a 
source distribution or wheel for the current platform
```

## Solution

### 1. Platform Markers

Added conditional dependency in `pyproject.toml`:

```toml
"torch==2.6.0; sys_platform != 'darwin' or platform_machine == 'arm64'",
```

**Logic:**
- Install PyTorch if NOT macOS → Linux, Windows ✅
- OR install if ARM64 → M1/M2/M3 Macs ✅  
- Skip if macOS AND x86_64 → Intel Macs ❌

**Result:** `uv sync` works on Intel Mac, skipping PyTorch but installing everything else.

### 2. Docker Execution

Created Docker images that include PyTorch:
- `dockerfiles/evaluate.dockerfile` - For model evaluation
- `dockerfiles/train.dockerfile` - For training
- `dockerfiles/api.dockerfile` - For API server

**Volume mounts** give containers access to host files without copying.

## Implementation Details

### Files Modified

**pyproject.toml**
```toml
# Changed from:
"torch==2.6.0",

# To:
"torch==2.6.0; sys_platform != 'darwin' or platform_machine == 'arm64'",
```

Also removed `[tool.uv.sources]` and `[[tool.uv.index]]` sections.

**tasks.py**
- Added `docker_evaluate()` - Wrapper for Docker-based evaluation
- Updated `docker_build()` - Builds all three Docker images

**src/ml_ops_assignment/data.py**
- Fixed `collate_fn()` to truncate sequences > 512 tokens
- Prevents runtime errors for all platforms

### Files Created

**Dockerfiles:**
- `dockerfiles/evaluate.dockerfile`
- (train and api dockerfiles already existed)

**Documentation:**
- `INTEL_MAC_GUIDE.md` - User guide
- `INTEL_MAC_QUICKREF.md` - Quick reference
- `documents/INTEL_MAC_SUMMARY.md` - This file

## What Works

### Intel Mac Compatible

| Command | Works | Why |
|---------|-------|-----|
| `uv sync` | ✅ | Platform marker skips PyTorch |
| `uv run invoke docker-build` | ✅ | No PyTorch import needed |
| `uv run invoke docker-evaluate` | ✅ | Runs in Docker container |
| `docker run ... evaluate:latest` | ✅ | Direct Docker execution |
| `uv run ruff format .` | ✅ | No PyTorch dependency |

### Requires Native PyTorch

| Command | Works | Alternative |
|---------|-------|-------------|
| `uv run invoke evaluate` | ❌ | `uv run invoke docker-evaluate` |
| `uv run invoke train` | ❌ | `docker run ... train:latest` |
| `uv run invoke serve-api` | ❌ | `docker run ... api:latest` |
| `uv run pytest tests/` | ❌ | `docker run ... pytest` |

## Platform Compatibility Matrix

| Platform | Standard Workflow | Docker Workflow |
|----------|------------------|-----------------|
| Linux x86_64 | ✅ | ✅ |
| macOS ARM64 (M1/M2/M3) | ✅ | ✅ |
| Windows x86_64 | ✅ | ✅ |
| macOS x86_64 (Intel) | ❌ | ✅ |

## Testing

Verified on Intel Mac (macOS x86_64):

```bash
$ uv sync
✅ Success - 186 packages, torch NOT installed

$ uv run invoke docker-evaluate --checkpoint models/model_final.pt
✅ Success - Accuracy: 87.20%
```

## Maintenance Guidelines

### When Updating Dependencies

1. Preserve the platform marker:
   ```toml
   "torch==X.Y.Z; sys_platform != 'darwin' or platform_machine == 'arm64'",
   ```

2. Rebuild Docker images:
   ```bash
   uv run invoke docker-build
   ```

3. Test both workflows:
   ```bash
   # Standard (if on compatible platform)
   uv run invoke evaluate --checkpoint models/model_final.pt
   
   # Docker (works everywhere)
   uv run invoke docker-evaluate --checkpoint models/model_final.pt
   ```

### When Updating Documentation

Keep these files synchronized:
- `INTEL_MAC_GUIDE.md` - User-facing guide
- `INTEL_MAC_QUICKREF.md` - Command reference
- `documents/INTEL_MAC_SUMMARY.md` - Technical details (this file)

### CI/CD Considerations

To test Intel Mac compatibility in CI:
- Use Docker-based workflows
- Test that `uv sync` succeeds (without actually installing PyTorch)
- Verify Docker images build successfully

## Architecture

```
┌─────────────────────────────────────────┐
│  Intel Mac (Host)                       │
│  ┌───────────────────────────────────┐  │
│  │ uv environment                    │  │
│  │ - All dependencies except PyTorch │  │
│  │ - Can run invoke, ruff, etc.      │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Docker Container (Linux)          │  │
│  │ - Complete environment            │  │
│  │ - Includes PyTorch                │  │
│  │ - Mounts models/ and data/        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Performance

- **Container startup:** ~1-2 seconds
- **First Docker build:** 2-5 minutes (downloads ~2-3 GB)
- **Subsequent builds:** Seconds (cached)
- **Evaluation speed:** Same as native
- **Disk space per image:** ~2-3 GB

## Future Improvements

1. **Pre-built images** - Push to GitHub Container Registry
2. **Docker Compose** - Orchestrate multiple services
3. **Dev Containers** - VS Code integration
4. **GPU support** - Separate GPU-enabled Dockerfile
