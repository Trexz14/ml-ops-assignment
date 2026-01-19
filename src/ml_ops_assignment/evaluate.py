"""
Evaluation script for trained text classification models.

This script loads a trained model checkpoint and evaluates it on the test set.
Usage: uv run python src/ml_ops_assignment/evaluate.py models/best_model.pt
"""

from pathlib import Path

from loguru import logger

from ml_ops_assignment.data import get_dataloaders
from ml_ops_assignment.model import evaluate, get_loss_function, load_config, load_model


def main(
    checkpoint: str = "models/model_final.pt",
    data_path: str = "data/processed",
    batch_size: int = 32,
    split: str = "test",
) -> None:
    """Evaluate a trained model on the specified data split."""
    checkpoint_path = Path(checkpoint)
    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    config_path = Path("configs/experiments/default.yaml")
    config_dict = load_config(config_path)

    # Load model
    model = load_model(checkpoint_path, config_path=config_path)
    device = next(model.parameters()).device

    # Load data
    logger.info(f"Loading {split} data from {data_path}")
    dataloader = get_dataloaders(Path(data_path), batch_size=batch_size, split=split)

    # Get loss function
    loss_fn = get_loss_function(config_dict["training"]["loss_function"])

    # Evaluate
    logger.info(f"Evaluating on {split} set...")
    results = evaluate(model, dataloader, loss_fn, device)

    # Print results
    logger.info("=" * 50)
    logger.info(f"Evaluation Results ({split} set):")
    logger.info(f"  Loss: {results['loss']:.4f}")
    logger.info(f"  Accuracy: {results['accuracy']:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "models/model_final.pt"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    main(checkpoint=checkpoint, split=split)
