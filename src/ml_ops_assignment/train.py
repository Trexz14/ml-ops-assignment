"""
Training script for the text classification model.

This script provides a simple command-line interface to train the model
using the configuration file. It imports and uses the train function from model.py.
"""

from dotenv import load_dotenv
load_dotenv()

import os 
import wandb

from pathlib import Path

import typer

from ml_ops_assignment.model import train

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
    typer.echo(f"Starting training with config: {config_path}")

    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    # Train the model
    train(config_path=config_path, checkpoint_path=checkpoint)

    typer.echo("Training completed successfully!")

    # ✅ FINISH WANDB HERE
    wandb.finish()


if __name__ == "__main__":
    app()
