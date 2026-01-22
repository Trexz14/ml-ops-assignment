from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
import yaml
from torch import nn
from torch.optim import Adam
from transformers import AutoModel

from ml_ops_assignment.data import get_dataloaders
from ml_ops_assignment.logging_config import get_logger

logger = get_logger(__name__)


class TextClassificationModel(nn.Module):
    """
    Text classification model for predicting text quality (0-2 scale).

    This model uses a pre-trained transformer as a feature extractor and adds
    a classification head for predicting text quality scores.

    Attributes:
        transformer: Pre-trained transformer model for text encoding
        dropout: Dropout layer for regularization
        classifier: Linear layer for classification
        num_labels: Number of output classes (3 for 0, 1, 2)
    """

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-mini",
        num_labels: int = 3,
        dropout: float = 0.1,
        hidden_size: int = 256,
    ):
        """
        Initialize the text classification model.

        Args:
            model_name: Name of the pre-trained transformer model from HuggingFace
            num_labels: Number of output classes (default: 3 for labels 0, 1, 2)
            dropout: Dropout rate for regularization (default: 0.1)
            hidden_size: Hidden size for the classifier head (default: 256)
        """
        super().__init__()

        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer_hidden_size = self.transformer.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(self.transformer_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)

        Returns:
            Logits for each class, shape (batch_size, num_labels)
        """
        # Get transformer outputs
        # Output shape: (batch_size, seq_length, hidden_size)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        # Shape: (batch_size, hidden_size)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)

        # Pass through hidden layer
        hidden = self.hidden_layer(pooled_output)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Classification logits
        # Shape: (batch_size, num_labels)
        logits = self.classifier(hidden)

        return logits


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.success(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def load_model(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> TextClassificationModel:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        config_path: Optional path to config file (will infer from checkpoint if not provided)
        device: Device to load model to (defaults to CPU)

    Returns:
        Loaded model ready for inference
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    if device is None:
        device = torch.device("cpu")

    # Load checkpoint
    logger.debug(f"Loading checkpoint to device: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or config file
    if "config" in checkpoint:
        model_config = checkpoint["config"]
        logger.debug("Using config from checkpoint")
    elif config_path is not None:
        config = load_config(config_path)
        model_config = config["model"]
        logger.debug(f"Using config from file: {config_path}")
    else:
        logger.error("Config not found in checkpoint and no config_path provided")
        raise ValueError("Config not found in checkpoint and no config_path provided")

    # Initialize model
    logger.info(f"Initializing model: {model_config['model_name']}")
    model = TextClassificationModel(
        model_name=model_config["model_name"],
        num_labels=model_config["num_labels"],
        dropout=model_config.get("dropout", 0.1),
        hidden_size=model_config.get("hidden_size", 256),
    )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.debug("Loaded model_state_dict from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        logger.debug("Loaded state dict directly from checkpoint")

    model.to(device)
    model.eval()
    logger.success(f"Model loaded successfully and set to eval mode on {device}")

    return model


def get_device(config: Dict) -> torch.device:
    """
    Get the appropriate device for training based on config and availability.

    Args:
        config: Configuration dictionary with device preference

    Returns:
        torch.device object for model placement
    """
    preferred_device = config.get("device", "cuda")

    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_loss_function(loss_name: str) -> nn.Module:
    """
    Get the loss function based on configuration.

    Args:
        loss_name: Name of the loss function

    Returns:
        Loss function module

    Raises:
        ValueError: If loss function name is not supported
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def evaluate(
    model: TextClassificationModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate `model` on `dataloader` and return loss and accuracy.

    Args:
        model: Trained model to evaluate
        dataloader: DataLoader for the evaluation split
        loss_fn: Loss function to compute evaluation loss
        device: Device to run evaluation on

    Returns:
        Dict with keys `loss` and `accuracy` (percentage)
    """
    logger.info("Starting model evaluation")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                logger.debug(f"Evaluation batch [{batch_idx+1}/{len(dataloader)}]")

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    logger.info(f"Evaluation complete: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    return {"loss": avg_loss, "accuracy": accuracy}


def train(
    config_path: Path = Path("configs/experiments/default.yaml"),
    checkpoint_path: Optional[Path] = None,
) -> TextClassificationModel:
    """
    Train the text classification model.

    This function handles the complete training loop including:
    - Loading configuration
    - Initializing model, optimizer, and loss function
    - Loading training and validation data
    - Training for specified epochs with validation
    - Saving checkpoints

    Args:
        config_path: Path to the configuration YAML file
        checkpoint_path: Optional path to load a checkpoint from

    Returns:
        Trained model
    """
    # Load configuration
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info(f"Configuration file: {config_path}")
    logger.info("=" * 60)

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Extract configuration sections
    model_config = config["model"]
    training_config = config["training"]

    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Training epochs: {training_config['num_epochs']}")
    logger.info(f"Batch size: {training_config['batch_size']}")
    logger.info(f"Learning rate: {training_config['learning_rate']}")

    # Set up device
    device = get_device(config)
    print(f"Using device: {device}")

    # Initialize model
    print(f"Initializing model: {model_config['model_name']}")
    model = TextClassificationModel(
        model_name=model_config["model_name"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        hidden_size=model_config["hidden_size"],
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.info("Starting training from scratch (no checkpoint provided)")

    model = model.to(device)

    # Initialize optimizer (Adam as specified)
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # Get loss function from config
    loss_fn = get_loss_function(training_config["loss_function"])

    # Load data
    print(f"Loading data from {training_config['data_path']}")
    data_path = Path(training_config["data_path"])
    train_loader = get_dataloaders(data_path, batch_size=training_config["batch_size"], split="train")

    # Validation can be toggled via config: training.do_validation (default: True)
    do_validation = training_config.get("do_validation", True)
    val_loader = None
    if do_validation:
        val_loader = get_dataloaders(data_path, batch_size=training_config["batch_size"], split="validation")

    # Training loop
    num_epochs = training_config["num_epochs"]
    gradient_clip = training_config["gradient_clip"]

    print(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Training configuration: {num_epochs} epochs, gradient_clip={gradient_clip}")

    if do_validation and val_loader is not None:
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    else:
        print(f"Training batches: {len(train_loader)}, Validation disabled")
        logger.info(f"Training batches: {len(train_loader)}, Validation disabled")

    for epoch in range(num_epochs):
        # Training phase
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Training phase started")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            # Input: (batch_size, seq_length)
            # Output: (batch_size, num_labels)
            logits = model(input_ids, attention_mask)

            # Calculate loss
            # loss_fn expects (batch_size, num_classes) and (batch_size,)
            loss = loss_fn(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # Update weights
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Validation phase (optional)
        if do_validation and val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            avg_val_loss = val_metrics["loss"]
            val_accuracy = val_metrics["accuracy"]
        else:
            avg_val_loss = 0.0
            val_accuracy = 0.0

        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"{'='*60}\n")

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] Summary: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%"
        )

        # Log to W&B if active
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": avg_val_loss,
                "val/accuracy": val_accuracy,
            })

        # Save checkpoint
        if (epoch + 1) % training_config["save_every"] == 0:
            checkpoint_dir = Path(training_config["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_file = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")
            logger.info(f"Checkpoint saved: {checkpoint_file}")

    # Save final model
    final_model_path = Path(training_config["checkpoint_dir"]) / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    logger.success(f"Training completed successfully! Final model saved to {final_model_path}")

    return model


if __name__ == "__main__":
    # Example usage: Train with default config
    model = train()

    # Example: Test model inference
    print("\nTesting model inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create dummy input (batch_size=2, seq_length=128)
    dummy_input_ids = torch.randint(0, 1000, (2, 128)).to(device)
    dummy_attention_mask = torch.ones(2, 128).to(device)

    with torch.no_grad():
        output = model(dummy_input_ids, dummy_attention_mask)

    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits:\n{output}")
    print(f"Predicted classes: {torch.argmax(output, dim=1)}")
