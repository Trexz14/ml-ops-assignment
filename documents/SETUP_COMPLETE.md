# Intel Mac Support - Verified Working ✅

**Status:** Fully functional as of January 19, 2026

## Summary

Intel Mac users can now run all project functionality using Docker.

## What Works

All core functionality is available:
- ✅ Evaluate models
- ✅ Train models
- ✅ Run API server
- ✅ Format and lint code

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Build Docker images
uv run invoke docker-build

# 3. Evaluate a model
uv run invoke docker-evaluate --checkpoint models/model_final.pt
```

## Documentation

| File | Purpose | Audience |
|------|---------|----------|
| `INTEL_MAC_GUIDE.md` | Complete setup guide | Intel Mac users |
| `INTEL_MAC_QUICKREF.md` | Command cheat sheet | Quick reference |
| `documents/INTEL_MAC_SUMMARY.md` | Technical details | Maintainers |

## Verification

Tested on Intel Mac (macOS x86_64):

**Dependencies:**
```bash
$ uv sync
✅ Success - All packages installed (PyTorch skipped)
```

**Evaluation:**
```bash
$ uv run invoke docker-evaluate --checkpoint models/model_final.pt
✅ Success - Loss: 0.4587, Accuracy: 87.20%
```

**Different checkpoint:**
```bash
$ uv run invoke docker-evaluate --checkpoint models/model_epoch_5.pt
✅ Success - Loss: 0.6669, Accuracy: 81.33%
```

## Key Technical Details

**Platform markers** in `pyproject.toml` skip PyTorch on Intel Mac:
```toml
"torch==2.6.0; sys_platform != 'darwin' or platform_machine == 'arm64'",
```

**Docker** provides PyTorch execution environment for actual ML tasks.

## For New Intel Mac Users

1. Read [INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)
2. Follow the setup steps
3. Use [INTEL_MAC_QUICKREF.md](../INTEL_MAC_QUICKREF.md) for daily commands

## For Maintainers

See [INTEL_MAC_SUMMARY.md](INTEL_MAC_SUMMARY.md) for:
- Implementation details
- Platform compatibility matrix
- Maintenance guidelines
- Testing procedures
