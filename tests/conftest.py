"""
Pytest configuration and shared fixtures for ML Ops Assignment tests.

This module provides:
- Common fixtures for model, data, and configuration
- Test setup and teardown utilities
- Marks and skip conditions
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_path(project_root):
    """Get the default configuration file path."""
    return project_root / "configs" / "experiments" / "default.yaml"


@pytest.fixture(scope="session")
def processed_data_path(project_root):
    """Get the processed data directory path."""
    return project_root / "data" / "processed"


@pytest.fixture(scope="session")
def models_path(project_root):
    """Get the models directory path."""
    return project_root / "models"


@pytest.fixture(scope="function")
def device():
    """Get the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="function")
def sample_input_ids():
    """Generate sample input_ids tensor."""
    return torch.randint(0, 1000, (4, 32))


@pytest.fixture(scope="function")
def sample_attention_mask():
    """Generate sample attention_mask tensor."""
    return torch.ones(4, 32, dtype=torch.long)


@pytest.fixture(scope="function")
def sample_labels():
    """Generate sample labels tensor."""
    return torch.randint(0, 3, (4,))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "requires_data: mark test as requiring processed data")
