#!/usr/bin/env python3
"""
Logging Demo Script

This script demonstrates the logging functionality without requiring torch.
It simulates various operations that would occur during training/evaluation.
"""

import time
from ml_ops_assignment.logging_config import get_logger

logger = get_logger(__name__)


def simulate_data_loading():
    """Simulate data loading operations."""
    logger.info("=" * 60)
    logger.info("Starting data loading simulation")
    logger.info("=" * 60)
    
    logger.info("Loading OneStop English dataset from Hugging Face")
    time.sleep(0.5)
    logger.success("Dataset loaded successfully with 2 splits")
    
    logger.info("Loading tokenizer for prajjwal1/bert-mini")
    time.sleep(0.3)
    logger.success("Tokenizer loaded successfully")
    
    logger.info("Starting tokenization (batched processing)")
    time.sleep(0.5)
    logger.success("Tokenization completed")
    
    logger.info("Merging existing splits and creating Train/Val/Test (80/10/10) split")
    logger.info("Merged 2 splits into 5500 total samples")
    logger.info("Split sizes -> Train: 4400, Val: 550, Test: 550")
    logger.debug("Sanity check - First sample is tensor: True")
    
    logger.info("Saving processed dataset to data/processed")
    logger.success("Data processing completed successfully! Dataset saved to data/processed")


def simulate_training():
    """Simulate model training."""
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("Configuration file: configs/experiments/default.yaml")
    logger.info("=" * 60)
    
    logger.info("Model: prajjwal1/bert-mini")
    logger.info("Training epochs: 10")
    logger.info("Batch size: 32")
    logger.info("Learning rate: 0.0001")
    
    logger.info("Using CPU device")
    logger.info("Starting training from scratch (no checkpoint provided)")
    
    logger.info("Training configuration: 10 epochs, gradient_clip=1.0")
    logger.info("Training batches: 138, Validation batches: 18")
    
    # Simulate 3 epochs
    for epoch in range(1, 4):
        logger.info(f"Epoch [{epoch}/10] - Training phase started")
        
        # Simulate some batches
        for batch in [10, 20, 30]:
            time.sleep(0.1)
            logger.debug(f"Batch [{batch}/138], Loss: 0.{95-epoch*10:02d}")
        
        # Epoch summary
        train_loss = 1.5 - epoch * 0.2
        train_acc = 40 + epoch * 10
        val_loss = 1.3 - epoch * 0.18
        val_acc = 45 + epoch * 10
        
        logger.info(
            f"Epoch [{epoch}/10] Summary: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )
        
        if epoch % 2 == 0:
            logger.info(f"Checkpoint saved: models/model_epoch_{epoch}.pt")
    
    logger.success("Training completed successfully! Final model saved to models/model_final.pt")


def simulate_evaluation():
    """Simulate model evaluation."""
    logger.info("=" * 60)
    logger.info("Evaluation Script Started")
    logger.info("Checkpoint: models/model_final.pt")
    logger.info("Data path: data/processed")
    logger.info("Batch size: 32")
    logger.info("Split: test")
    logger.info("=" * 60)
    
    logger.info("Loading model from models/model_final.pt")
    time.sleep(0.3)
    logger.success("Configuration loaded successfully from configs/experiments/default.yaml")
    
    logger.info("Loading dataset from data/processed for split 'test'")
    logger.info("DataLoader created: 18 batches, batch_size=32, shuffle=False")
    
    logger.info("Starting model evaluation")
    
    for batch_idx in [20, 40]:
        time.sleep(0.1)
        logger.debug(f"Evaluation batch [{batch_idx}/18]")
    
    logger.info("Evaluation complete: Loss=0.3245, Accuracy=87.50%")
    
    logger.info("=" * 50)
    logger.info("Evaluation Results (test set):")
    logger.info("  Loss: 0.3245")
    logger.info("  Accuracy: 87.50%")
    logger.info("=" * 50)
    logger.success("Evaluation completed successfully")


def simulate_api():
    """Simulate API operations."""
    logger.info("=" * 60)
    logger.info("FastAPI Application Starting Up")
    logger.info("=" * 60)
    
    logger.info("Checkpoint path: models/model_final.pt")
    logger.info("Config path: configs/experiments/default.yaml")
    
    logger.success("Configuration loaded successfully from configs/experiments/default.yaml")
    logger.info("Checkpoint found at models/model_final.pt")
    logger.info("Using device: cpu")
    logger.success("Model loaded successfully")
    
    logger.info("Loading tokenizer: prajjwal1/bert-mini")
    logger.success("Tokenizer loaded successfully")
    logger.success("Application startup complete")
    
    # Simulate some API requests
    logger.debug("Health check endpoint called")
    
    logger.info("Prediction request received for text: 'This is a sample text for classificatio...'")
    logger.debug("Tokenizing input with max_length=512")
    logger.debug("Running model inference")
    logger.info("Prediction complete: label=1, class_name=Intermediate")
    
    logger.info("Prediction request received for text: 'The weather is nice today...'")
    logger.debug("Tokenizing input with max_length=512")
    logger.debug("Running model inference")
    logger.info("Prediction complete: label=0, class_name=Elementary")
    
    logger.info("Shutting down application...")
    logger.success("Application shutdown complete")


def simulate_error():
    """Simulate an error scenario."""
    logger.info("Demonstrating error logging...")
    
    try:
        # Simulate an error
        raise ValueError("This is a simulated error for demonstration")
    except ValueError as e:
        logger.error(f"Caught an error: {e}")
        logger.error("This error is logged to both app.log and errors.log")


def main():
    """Run the logging demo."""
    print("\n" + "=" * 70)
    print("🎯 LOGGING DEMO - Simulating ML Pipeline Operations")
    print("=" * 70)
    print("\nThis demo simulates logging from various components:")
    print("  1. Data Loading")
    print("  2. Model Training")
    print("  3. Model Evaluation")
    print("  4. API Server")
    print("  5. Error Handling")
    print("\nWatch the console output and check logs/app.log and logs/errors.log")
    print("=" * 70 + "\n")
    
    input("Press ENTER to start the demo...")
    
    print("\n" + "🔹" * 35)
    print("1️⃣  SIMULATING DATA LOADING")
    print("🔹" * 35 + "\n")
    simulate_data_loading()
    
    input("\nPress ENTER to continue to training simulation...")
    
    print("\n" + "🔹" * 35)
    print("2️⃣  SIMULATING MODEL TRAINING")
    print("🔹" * 35 + "\n")
    simulate_training()
    
    input("\nPress ENTER to continue to evaluation simulation...")
    
    print("\n" + "🔹" * 35)
    print("3️⃣  SIMULATING MODEL EVALUATION")
    print("🔹" * 35 + "\n")
    simulate_evaluation()
    
    input("\nPress ENTER to continue to API simulation...")
    
    print("\n" + "🔹" * 35)
    print("4️⃣  SIMULATING API SERVER")
    print("🔹" * 35 + "\n")
    simulate_api()
    
    input("\nPress ENTER to see error logging...")
    
    print("\n" + "🔹" * 35)
    print("5️⃣  SIMULATING ERROR LOGGING")
    print("🔹" * 35 + "\n")
    simulate_error()
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\n📁 Check the log files:")
    print("   • logs/app.log    - All log messages (DEBUG, INFO, WARNING, ERROR)")
    print("   • logs/errors.log - Only ERROR and CRITICAL messages")
    print("\n💡 Try these commands:")
    print("   tail -f logs/app.log        # Watch logs in real-time")
    print("   grep 'ERROR' logs/app.log   # Find all errors")
    print("   cat logs/errors.log         # View error log")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
