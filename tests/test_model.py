"""
Unit tests for model module (M16).

This module tests:
- TextClassificationModel class construction and forward pass
- Output shape verification
- Configuration loading
- Device selection
- Loss function creation
- Model evaluation function
"""
import pytest
import torch
from pathlib import Path

from ml_ops_assignment.model import (
    TextClassificationModel,
    load_config,
    get_device,
    get_loss_function,
    evaluate,
)


class TestTextClassificationModel:
    """Tests for the TextClassificationModel class."""

    @pytest.fixture
    def config(self):
        """Fixture for model configuration."""
        return {
            "model_name": "prajjwal1/bert-mini",
            "num_labels": 3,
            "hidden_size": 256,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, config):
        """Fixture for model instance."""
        return TextClassificationModel(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        )

    @pytest.fixture
    def sample_batch(self):
        """Fixture for sample input batch."""
        batch_size = 4
        seq_length = 32
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
        }

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_returns_tensor(self, model, sample_batch):
        """Test that forward pass returns a tensor."""
        model.eval()
        with torch.no_grad():
            output = model(sample_batch["input_ids"], sample_batch["attention_mask"])
        assert torch.is_tensor(output)

    def test_model_output_shape(self, model, sample_batch, config):
        """Test that output has correct shape (batch_size, num_labels)."""
        model.eval()
        with torch.no_grad():
            output = model(sample_batch["input_ids"], sample_batch["attention_mask"])
        batch_size = sample_batch["input_ids"].shape[0]
        assert output.shape == (batch_size, config["num_labels"])

    def test_model_output_dtype(self, model, sample_batch):
        """Test that model outputs float tensor."""
        model.eval()
        with torch.no_grad():
            output = model(sample_batch["input_ids"], sample_batch["attention_mask"])
        assert output.dtype == torch.float32

    def test_model_different_batch_sizes(self, model, config):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, 32))
            attention_mask = torch.ones(batch_size, 32, dtype=torch.long)
            model.eval()
            with torch.no_grad():
                output = model(input_ids, attention_mask)
            assert output.shape == (batch_size, config["num_labels"])

    def test_model_different_sequence_lengths(self, model, config):
        """Test model with different sequence lengths."""
        for seq_length in [16, 64, 128]:
            input_ids = torch.randint(0, 1000, (4, seq_length))
            attention_mask = torch.ones(4, seq_length, dtype=torch.long)
            model.eval()
            with torch.no_grad():
                output = model(input_ids, attention_mask)
            assert output.shape == (4, config["num_labels"])

    def test_model_has_transformer(self, model):
        """Test that model has a transformer component."""
        assert hasattr(model, "transformer")

    def test_model_has_classifier(self, model):
        """Test that model has a classifier component."""
        assert hasattr(model, "classifier")

    def test_model_trainable_parameters(self, model):
        """Test that model has trainable parameters."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_model_gradient_flow(self, model, sample_batch):
        """Test that gradients flow through the model."""
        model.train()
        output = model(sample_batch["input_ids"], sample_batch["attention_mask"])
        loss = output.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestLoadConfig:
    """Tests for the load_config function."""

    @pytest.fixture
    def config_path(self):
        """Fixture for default config path."""
        return Path("configs/experiments/default.yaml")

    def test_load_config_returns_dict(self, config_path):
        """Test that load_config returns a dictionary."""
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(str(config_path))
        assert isinstance(config, dict)

    def test_load_config_has_model_section(self, config_path):
        """Test that config contains model section."""
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(str(config_path))
        assert "model" in config
        assert "model_name" in config["model"]

    def test_load_config_has_training_section(self, config_path):
        """Test that config contains training section."""
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(str(config_path))
        assert "training" in config
        assert "batch_size" in config["training"]

    def test_load_config_has_device(self, config_path):
        """Test that config contains device setting."""
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(str(config_path))
        assert "device" in config

    def test_load_config_model_params(self, config_path):
        """Test that config has required model parameters."""
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(str(config_path))
        assert "num_labels" in config["model"]
        assert "dropout" in config["model"]

    def test_load_config_invalid_path_raises_error(self):
        """Test that invalid config path raises an error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/path/config.yaml")


class TestGetDevice:
    """Tests for the get_device function."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a torch.device object."""
        config = {"device": "cpu"}
        device = get_device(config)
        assert isinstance(device, torch.device)

    def test_get_device_valid_type(self):
        """Test that device type is cpu, cuda, or mps."""
        config = {"device": "cpu"}
        device = get_device(config)
        assert device.type in ["cpu", "cuda", "mps"]

    def test_get_device_respects_cpu_config(self):
        """Test that get_device returns CPU when configured."""
        config = {"device": "cpu"}
        device = get_device(config)
        assert device.type == "cpu"

    def test_get_device_default_to_cpu(self):
        """Test that get_device defaults to CPU when CUDA unavailable."""
        config = {"device": "cuda"}
        device = get_device(config)
        # Should return cpu if cuda not available, or cuda if available
        assert device.type in ["cpu", "cuda"]


class TestGetLossFunction:
    """Tests for the get_loss_function function."""

    def test_get_loss_function_cross_entropy_returns_module(self):
        """Test that get_loss_function returns a loss module for cross_entropy."""
        loss_fn = get_loss_function("cross_entropy")
        assert isinstance(loss_fn, torch.nn.Module)

    def test_get_loss_function_computes_loss(self):
        """Test that loss function computes a valid loss."""
        loss_fn = get_loss_function("cross_entropy")
        predictions = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(predictions, targets)
        assert torch.is_tensor(loss)
        assert loss.dim() == 0  # Scalar loss

    def test_get_loss_function_loss_non_negative(self):
        """Test that computed loss is non-negative."""
        loss_fn = get_loss_function("cross_entropy")
        predictions = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(predictions, targets)
        assert loss.item() >= 0

    def test_get_loss_function_invalid_raises_error(self):
        """Test that invalid loss function raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported loss function"):
            get_loss_function("invalid_loss")


class TestEvaluate:
    """Tests for the evaluate function."""

    @pytest.fixture
    def config(self):
        """Fixture for model configuration."""
        return {
            "model_name": "prajjwal1/bert-mini",
            "num_labels": 3,
            "hidden_size": 256,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, config):
        """Fixture for model instance."""
        return TextClassificationModel(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        )

    @pytest.fixture
    def mock_dataloader(self):
        """Fixture for mock dataloader."""
        batches = [
            {
                "input_ids": torch.randint(0, 1000, (4, 32)),
                "attention_mask": torch.ones(4, 32, dtype=torch.long),
                "label": torch.randint(0, 3, (4,)),
            }
            for _ in range(3)
        ]
        return batches

    def test_evaluate_returns_dict(self, model, mock_dataloader):
        """Test that evaluate returns a dictionary."""
        device = torch.device("cpu")
        model.to(device)
        loss_fn = get_loss_function("cross_entropy")
        result = evaluate(model, mock_dataloader, loss_fn, device)
        assert isinstance(result, dict)

    def test_evaluate_contains_loss(self, model, mock_dataloader):
        """Test that evaluation result contains loss."""
        device = torch.device("cpu")
        model.to(device)
        loss_fn = get_loss_function("cross_entropy")
        result = evaluate(model, mock_dataloader, loss_fn, device)
        assert "loss" in result

    def test_evaluate_contains_accuracy(self, model, mock_dataloader):
        """Test that evaluation result contains accuracy."""
        device = torch.device("cpu")
        model.to(device)
        loss_fn = get_loss_function("cross_entropy")
        result = evaluate(model, mock_dataloader, loss_fn, device)
        assert "accuracy" in result

    def test_evaluate_loss_non_negative(self, model, mock_dataloader):
        """Test that evaluation loss is non-negative."""
        device = torch.device("cpu")
        model.to(device)
        loss_fn = get_loss_function("cross_entropy")
        result = evaluate(model, mock_dataloader, loss_fn, device)
        assert result["loss"] >= 0

    def test_evaluate_accuracy_in_valid_range(self, model, mock_dataloader):
        """Test that accuracy is between 0 and 100."""
        device = torch.device("cpu")
        model.to(device)
        loss_fn = get_loss_function("cross_entropy")
        result = evaluate(model, mock_dataloader, loss_fn, device)
        assert 0 <= result["accuracy"] <= 100


class TestModelIntegration:
    """Integration tests for model training workflow."""

    @pytest.fixture
    def config(self):
        """Fixture for model configuration."""
        return {
            "model_name": "prajjwal1/bert-mini",
            "num_labels": 3,
            "hidden_size": 256,
            "dropout": 0.1,
        }

    @pytest.fixture
    def model(self, config):
        """Fixture for model instance."""
        return TextClassificationModel(
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        )

    def test_model_save_and_load(self, model, tmp_path):
        """Test saving and loading model weights."""
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        assert save_path.exists()
        loaded_state = torch.load(save_path, weights_only=True)
        assert isinstance(loaded_state, dict)
        assert len(loaded_state) > 0

    def test_model_to_device(self, model):
        """Test moving model to CPU device."""
        device = torch.device("cpu")
        model.to(device)
        first_param = next(model.parameters())
        assert first_param.device.type == device.type

    def test_model_train_eval_modes(self, model):
        """Test switching between train and eval modes."""
        model.train()
        assert model.training
        model.eval()
        assert not model.training
