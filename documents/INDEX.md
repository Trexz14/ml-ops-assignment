# Intel Mac Documentation

Quick navigation to Intel Mac setup documentation.

## For Users

### [INTEL_MAC_GUIDE.md](../INTEL_MAC_GUIDE.md)
**Complete setup guide for Intel Mac users**

What's inside:
- Prerequisites checklist
- Step-by-step setup instructions
- Daily usage commands
- Troubleshooting common issues
- Command explanations

Start here if you're setting up for the first time.

---

### [INTEL_MAC_QUICKREF.md](../INTEL_MAC_QUICKREF.md)
**Quick command reference**

What's inside:
- One-time setup commands
- Evaluate, train, and API commands
- Formatting and linting
- Quick troubleshooting

Use this for daily work once you're set up.

## For Maintainers

### [INTEL_MAC_SUMMARY.md](INTEL_MAC_SUMMARY.md)
**Technical implementation details**

What's inside:
- Problem and solution explanation
- Platform marker implementation
- Docker architecture
- Compatibility matrix
- Maintenance guidelines
- Testing procedures

Read this to understand how the solution works.

---

### [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
**Verification and status**

What's inside:
- Confirmation that Intel Mac support is working
- Test results
- Quick start summary
- Documentation overview

Quick reference for what's been implemented.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Build Docker images
uv run invoke docker-build

# 3. Evaluate a model
uv run invoke docker-evaluate --checkpoint models/model_final.pt
```

## File Structure

```
├── INTEL_MAC_GUIDE.md          # User guide (start here)
├── INTEL_MAC_QUICKREF.md       # Quick reference
└── documents/
    ├── INDEX.md                # This file
    ├── INTEL_MAC_SUMMARY.md    # Technical details (maintainers)
    └── SETUP_COMPLETE.md       # Status confirmation
```

## Which File Do I Need?

| I want to... | Read this |
|-------------|-----------|
| Set up my Intel Mac | `INTEL_MAC_GUIDE.md` |
| Find a command quickly | `INTEL_MAC_QUICKREF.md` |
| Understand the implementation | `documents/INTEL_MAC_SUMMARY.md` |
| Verify it's working | `documents/SETUP_COMPLETE.md` |
| Contribute/maintain | `documents/INTEL_MAC_SUMMARY.md` |
