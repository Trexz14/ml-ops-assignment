"""
Unit tests for data processing module (M16).

This module tests:
- Data loading from Hugging Face
- Collate function for batch processing
- DataLoader creation and functionality
- Data format and shape validation
"""
import pytest
import torch
from pathlib import Path

from ml_ops_assignment.data import load_data, collate_fn, get_dataloaders


class TestLoadData:
    """Tests for the load_data function."""

    def test_load_data_returns_dataset(self):
        """Test that load_data returns a dataset object."""
        dataset = load_data()
        assert dataset is not None
        assert hasattr(dataset, "keys") or hasattr(dataset, "__len__")

    def test_load_data_has_expected_splits(self):
        """Test that loaded dataset has expected splits."""
        dataset = load_data()
        assert "train" in dataset or "test" in dataset

    def test_load_data_has_text_column(self):
        """Test that dataset contains 'text' column for NLP processing."""
        dataset = load_data()
        first_split = list(dataset.keys())[0]
        assert "text" in dataset[first_split].column_names

    def test_load_data_has_label_column(self):
        """Test that dataset contains 'label' column for classification."""
        dataset = load_data()
        first_split = list(dataset.keys())[0]
        assert "label" in dataset[first_split].column_names


class TestCollateFn:
    """Tests for the custom collate function."""

    def test_collate_fn_returns_dict(self):
        """Test that collate_fn returns a dictionary with expected keys."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "label": 0},
            {"input_ids": torch.tensor([4, 5, 6]), "attention_mask": torch.tensor([1, 1, 1]), "label": 1},
        ]
        result = collate_fn(batch)
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "label" in result

    def test_collate_fn_pads_sequences(self):
        """Test that collate_fn pads sequences of different lengths."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "label": 0},
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1]), "label": 1},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape[1] == 3
        assert result["attention_mask"].shape[1] == 3

    def test_collate_fn_returns_tensors(self):
        """Test that collate_fn returns PyTorch tensors."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "label": 0},
            {"input_ids": torch.tensor([4, 5, 6]), "attention_mask": torch.tensor([1, 1, 1]), "label": 1},
        ]
        result = collate_fn(batch)
        assert torch.is_tensor(result["input_ids"])
        assert torch.is_tensor(result["attention_mask"])
        assert torch.is_tensor(result["label"])

    def test_collate_fn_preserves_batch_size(self):
        """Test that collate_fn preserves the batch size."""
        batch_size = 4
        batch = [
            {"input_ids": torch.tensor([i, i + 1]), "attention_mask": torch.tensor([1, 1]), "label": i % 3}
            for i in range(batch_size)
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape[0] == batch_size
        assert result["attention_mask"].shape[0] == batch_size
        assert result["label"].shape[0] == batch_size

    def test_collate_fn_labels_dtype(self):
        """Test that labels are converted to proper tensor dtype."""
        batch = [
            {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]), "label": 0},
            {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 1]), "label": 2},
        ]
        result = collate_fn(batch)
        assert result["label"].dtype == torch.int64


class TestGetDataloaders:
    """Tests for the get_dataloaders function."""

    @pytest.fixture
    def processed_data_path(self):
        """Fixture for processed data path."""
        return Path("data/processed")

    def test_get_dataloaders_returns_dataloader(self, processed_data_path):
        """Test that get_dataloaders returns a DataLoader object."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=8, split="train")
        assert dataloader is not None
        from torch.utils.data import DataLoader

        assert isinstance(dataloader, DataLoader)

    def test_get_dataloaders_train_split(self, processed_data_path):
        """Test loading the train split."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=8, split="train")
        assert len(dataloader) > 0

    def test_get_dataloaders_validation_split(self, processed_data_path):
        """Test loading the validation split."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=8, split="validation")
        assert len(dataloader) > 0

    def test_get_dataloaders_test_split(self, processed_data_path):
        """Test loading the test split."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=8, split="test")
        assert len(dataloader) > 0

    def test_get_dataloaders_invalid_split_raises_error(self, processed_data_path):
        """Test that invalid split name raises ValueError."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        with pytest.raises(ValueError, match="not found in dataset"):
            get_dataloaders(processed_data_path, batch_size=8, split="invalid_split")

    def test_get_dataloaders_batch_format(self, processed_data_path):
        """Test that batches have correct format."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=4, split="train")
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "label" in batch

    def test_get_dataloaders_batch_tensor_shapes(self, processed_data_path):
        """Test that batch tensors have expected shapes."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        batch_size = 4
        dataloader = get_dataloaders(processed_data_path, batch_size=batch_size, split="train")
        batch = next(iter(dataloader))
        assert batch["input_ids"].dim() == 2
        assert batch["attention_mask"].dim() == 2
        assert batch["label"].dim() == 1
        assert batch["input_ids"].shape[0] <= batch_size

    def test_get_dataloaders_respects_batch_size(self, processed_data_path):
        """Test that dataloader respects batch size parameter."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        batch_size = 8
        dataloader = get_dataloaders(processed_data_path, batch_size=batch_size, split="train")
        batch = next(iter(dataloader))
        assert batch["input_ids"].shape[0] <= batch_size


class TestDataIntegration:
    """Integration tests for the data pipeline."""

    @pytest.fixture
    def processed_data_path(self):
        """Fixture for processed data path."""
        return Path("data/processed")

    def test_full_data_pipeline_iteration(self, processed_data_path):
        """Test iterating through entire dataloader."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=32, split="train")
        total_samples = 0
        for batch in dataloader:
            total_samples += batch["label"].shape[0]
        assert total_samples > 0

    def test_data_labels_in_valid_range(self, processed_data_path):
        """Test that all labels are in expected range [0, 1, 2]."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=32, split="train")
        for batch in dataloader:
            labels = batch["label"]
            assert labels.min() >= 0
            assert labels.max() <= 2

    def test_input_ids_non_negative(self, processed_data_path):
        """Test that input_ids are non-negative (valid token IDs)."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=32, split="train")
        batch = next(iter(dataloader))
        assert batch["input_ids"].min() >= 0

    def test_attention_mask_binary(self, processed_data_path):
        """Test that attention mask contains only 0s and 1s."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")
        dataloader = get_dataloaders(processed_data_path, batch_size=32, split="train")
        batch = next(iter(dataloader))
        unique_values = torch.unique(batch["attention_mask"])
        assert all(v in [0, 1] for v in unique_values.tolist())
