# Intel Mac Documentation Index

Quick navigation to all Intel Mac related documentation.

## 📚 Main Documentation Files

### For Users

1. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)**
   - ⭐ **START HERE** - Overview of the complete solution
   - What was accomplished and why
   - Quick test commands
   - Before vs after comparison

2. **[INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)**
   - 📖 Complete guide for Intel Mac users
   - Prerequisites and setup
   - Detailed explanations of Docker commands
   - Comprehensive troubleshooting
   - Comparison table between workflows

3. **[INTEL_MAC_QUICKREF.md](../INTEL_MAC_QUICKREF.md)**
   - 🚀 Quick reference cheat sheet
   - Copy-paste ready commands
   - Most common tasks
   - Quick troubleshooting tips

4. **[QUICKSTART.md](../QUICKSTART.md)**
   - General quick start guide (all users)
   - Includes Intel Mac specific sections
   - Standard workflow + Docker alternatives

### For Maintainers

5. **[INTEL_MAC_SUMMARY.md](INTEL_MAC_SUMMARY.md)**
   - 🔧 Technical summary of all changes
   - Files created and modified
   - Architecture decisions
   - Recommendations for future development

## 🗺️ Document Map

```
├── README.md                      # Project README with Intel Mac warning
├── QUICKSTART.md                  # Quick start guide (updated)
├── INTEL_MAC_GUIDE.md            # Comprehensive Intel Mac guide
├── INTEL_MAC_QUICKREF.md         # Quick reference commands
└── documents/
    ├── INDEX.md                  # This file
    ├── SETUP_COMPLETE.md         # Solution overview & test results
    └── INTEL_MAC_SUMMARY.md      # Technical documentation
```

## 🎯 Which Document Should I Read?

### "I just want to run evaluation quickly"
→ **[INTEL_MAC_QUICKREF.md](../INTEL_MAC_QUICKREF.md)**

### "I'm new and want to understand the full setup"
→ **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** then **[INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)**

### "I need to troubleshoot an issue"
→ **[INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)** (Troubleshooting section)

### "I'm contributing to the project"
→ **[INTEL_MAC_SUMMARY.md](INTEL_MAC_SUMMARY.md)**

### "I want the standard workflow"
→ **[QUICKSTART.md](../QUICKSTART.md)**

## 📋 Quick Command Reference

See [INTEL_MAC_QUICKREF.md](../INTEL_MAC_QUICKREF.md) for detailed commands.

**Build:**
```bash
docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile
```

**Evaluate:**
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  evaluate:latest models/model_final.pt test
```

## 🔄 Related Files

### Docker Files
- `dockerfiles/evaluate.dockerfile` - Evaluation Docker image
- `dockerfiles/train.dockerfile` - Training Docker image
- `dockerfiles/api.dockerfile` - API Docker image

### Code Changes
- `tasks.py` - Added Docker invoke tasks
- `src/ml_ops_assignment/data.py` - Fixed collate_fn bug

### Configuration
- `pyproject.toml` - Dependencies (including torch==2.6.0)
- `configs/experiments/default.yaml` - Training configuration

## 💡 Key Concepts

- **Docker** - Containerization platform providing consistent environment
- **Volume Mounts** - `-v` flag maps local directories into container
- **Image** - Template for containers (build once, run many times)
- **Container** - Running instance of an image (ephemeral)

## 🎓 Learning Resources

- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **Docker Tutorial**: https://docs.docker.com/get-started/
- **UV Package Manager**: https://github.com/astral-sh/uv

## ✅ Checklist

Use this to ensure everything is set up:

- [ ] Docker Desktop installed and running
- [ ] Models downloaded (`dvc pull`)
- [ ] Docker image built (`docker build ...`)
- [ ] Evaluation tested successfully
- [ ] Read relevant documentation

## 🆘 Getting Help

1. Check the Troubleshooting section in [INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)
2. Review error messages in the terminal
3. Ensure Docker Desktop is running
4. Verify models and data are downloaded
5. Try rebuilding the Docker image

## 📊 Success Metrics

Your setup is working if you can:
- ✅ Build Docker images without errors
- ✅ Run evaluation and see accuracy/loss results
- ✅ Evaluate different model checkpoints
- ✅ Access logs and outputs

---

**Last Updated:** January 19, 2026
**Solution Status:** ✅ Working and Tested
**Platform:** Intel Mac (macOS x86_64)
