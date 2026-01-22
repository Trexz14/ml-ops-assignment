"""
Training script for the text classification model.

This script provides a simple command-line interface to train the model
using the configuration file. It imports and uses the train function from model.py.
"""

import os
from pathlib import Path

import typer
import wandb
from dotenv import load_dotenv

from ml_ops_assignment.model import train
from ml_ops_assignment.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/experiments/default.yaml"), help="Path to the configuration YAML file"
    ),
    checkpoint: Path = typer.Option(None, help="Path to a checkpoint file to resume training from"),
):
    """
    Train the text classification model.

    Args:
        config_path: Path to the YAML configuration file
        checkpoint: Optional path to resume training from a checkpoint
    """
    logger.info("=" * 70)
    logger.info("Training Script Started")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Checkpoint: {checkpoint if checkpoint else 'None (training from scratch)'}")
    logger.info("=" * 70)

    typer.echo(f"Starting training with config: {config_path}")

    # Initialize W&B
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")

    if wandb_project:
        logger.info(f"Initializing W&B: project={wandb_project}, entity={wandb_entity or 'default'}")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
        )
        logger.success("W&B initialized successfully")
    else:
        logger.warning("W&B PROJECT not found in environment variables. Skipping W&B initialization.")

    # Train the model
    train(config_path=config_path, checkpoint_path=checkpoint)

    typer.echo("Training completed successfully!")
    logger.success("Training completed successfully!")

    # ✅ FINISH WANDB HERE
    if wandb.run is not None:
        logger.info("Finishing W&B run")
        wandb.finish()
        logger.success("W&B run finished")
    else:
        logger.debug("No active W&B run to finish")


if __name__ == "__main__":
    app()
