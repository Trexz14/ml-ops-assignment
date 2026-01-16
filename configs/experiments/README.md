# Experiment Configurations

This directory contains YAML configuration files for training the text classification model.

## Configuration Structure

Each configuration file should contain the following sections:

### `model`
- `model_name`: Pre-trained transformer model from HuggingFace (e.g., "prajjwal1/bert-mini")
- `num_labels`: Number of output classes (3 for labels 0, 1, 2)
- `dropout`: Dropout rate for regularization (typically 0.1-0.3)
- `hidden_size`: Size of the hidden layer in the classification head

### `training`
- `optimizer`: Optimizer type (currently supports "adam")
- `learning_rate`: Learning rate for the optimizer (e.g., 2e-5)
- `weight_decay`: Weight decay for L2 regularization
- `loss_function`: Loss function to use (currently supports "cross_entropy")
- `batch_size`: Number of samples per batch
- `num_epochs`: Number of training epochs
- `gradient_clip`: Maximum gradient norm for gradient clipping
- `data_path`: Path to the processed dataset
- `save_every`: Save checkpoint every N epochs
- `checkpoint_dir`: Directory to save model checkpoints

### `device`
- Device preference: "cuda", "mps", or "cpu" (will auto-detect if not available)

## Usage

To train with a specific configuration:

```bash
uv run python -m ml_ops_assignment.train --config-path configs/experiments/your_config.yaml
```

Or using the train.py command:

```bash
uv run ml_ops_assignment/train.py --config-path configs/experiments/your_config.yaml
```

## Creating Custom Configurations

1. Copy `default.yaml` to a new file
2. Modify the parameters as needed
3. Run training with your custom config

Example configurations you might want to create:
- `fast_experiment.yaml` - Smaller model, fewer epochs for quick testing
- `production.yaml` - Best hyperparameters for production training
- `gpu_optimized.yaml` - Settings optimized for GPU training with larger batch sizes
