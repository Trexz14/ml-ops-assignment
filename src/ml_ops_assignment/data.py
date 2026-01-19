import torch
from pathlib import Path
import typer
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from ml_ops_assignment.logging_config import get_logger

logger = get_logger(__name__)
app = typer.Typer()


def load_data():
    """
    Load the OneStop English dataset from Hugging Face.

    Returns:
        DatasetDict: A dictionary containing the 'train' split of the dataset.
    """
    logger.info("Loading OneStop English dataset from Hugging Face")
    try:
        dataset = load_dataset("SetFit/onestop_english")
        logger.success(f"Dataset loaded successfully with {len(dataset)} splits")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


@app.command()
def process(
    output_folder: Path = typer.Argument(Path("data/processed"), help="Path to save processed data"),
    model_name: str = typer.Argument("prajjwal1/bert-mini", help="Model name for tokenizer"),
):
    """
    Load raw data, tokenize it, and save the processed dataset to disk.

    This function performs the following steps:
    1. Loads the raw 'onestop_english' dataset.
    2. Initializes a tokenizer for the specified model.
    3. Tokenizes the text data.
    4. Splits the data into Train (80%), Validation (10%), and Test (10%).
    5. Sets the dataset format to PyTorch tensors.
    6. Saves the processed dataset to the specified output folder for DVC tracking.

    Args:
        output_folder (Path): Directory where the processed dataset will be saved.
        model_name (str): The Hugging Face model checkpoint to use for tokenization.
    """
    logger.info("=" * 60)
    logger.info("Starting data processing pipeline")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Model name: {model_name}")
    logger.info("=" * 60)

    typer.echo("Loading data...")
    dataset = load_data()

    typer.echo(f"Loading tokenizer for {model_name}...")
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.success("Tokenizer loaded successfully")

    def tokenize_function(examples):
        """Tokenize the text column with padding and truncation."""
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    typer.echo("Tokenizing data...")
    # Batched tokenization
    logger.info("Starting tokenization (batched processing)")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    logger.success("Tokenization completed")

    # ---------------------------------------------------------
    # Splitting Strategy (80/10/10):
    # The dataset comes pre-split for few-shot learning (34% train, 66% test)
    # We need to merge them and create our own standard 80/10/10 split
    # ---------------------------------------------------------
    typer.echo("Merging existing splits and creating Train/Val/Test (80/10/10)...")
    logger.info("Merging existing splits and creating Train/Val/Test (80/10/10) split")

    # Concatenate all existing splits into one dataset
    from datasets import concatenate_datasets

    if isinstance(tokenized_datasets, dict) or hasattr(tokenized_datasets, "keys"):
        # It's a DatasetDict - merge all splits
        all_splits = list(tokenized_datasets.values())
        full_ds = concatenate_datasets(all_splits)
        typer.echo(f"Merged {len(all_splits)} splits into {len(full_ds)} total samples")
        logger.info(f"Merged {len(all_splits)} splits into {len(full_ds)} total samples")
    else:
        # It's already a single Dataset
        full_ds = tokenized_datasets
        logger.info(f"Using single dataset with {len(full_ds)} samples")

    # Now do our 80/10/10 split
    # First split: 10% for Test
    logger.info("Splitting data: First split for test set (10%)")
    train_val_test = full_ds.train_test_split(test_size=0.1, seed=42)
    test_ds = train_val_test["test"]
    train_val_ds = train_val_test["train"]  # This is 90% of original

    # Second split: 1/9 of the 90% = 10% of original for Validation
    logger.info("Splitting data: Second split for validation set (10%)")
    train_val_split = train_val_ds.train_test_split(test_size=1 / 9, seed=42)
    train_ds = train_val_split["train"]  # 8/9 of 90% = 80% of original
    val_ds = train_val_split["test"]  # 1/9 of 90% = 10% of original

    from datasets import DatasetDict

    final_dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    # Set format for PyTorch
    logger.info("Setting dataset format to PyTorch tensors")
    final_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Sanity Check
    typer.echo(
        f"Split sizes -> Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}"
    )
    logger.info(
        f"Split sizes -> Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}"
    )
    sample_tensor = final_dataset["train"][0]["input_ids"]
    typer.echo(f"Sanity Check - Is Tensor? {torch.is_tensor(sample_tensor)}")
    logger.debug(f"Sanity check - First sample is tensor: {torch.is_tensor(sample_tensor)}")

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save to disk
    typer.echo(f"Saving processed data to {output_folder}...")
    logger.info(f"Saving processed dataset to {output_folder}")
    final_dataset.save_to_disk(output_folder)
    typer.echo("Done!")
    logger.success(f"Data processing completed successfully! Dataset saved to {output_folder}")


def collate_fn(batch):
    """Custom collate function to pad sequences to the same length in a batch."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    # Pad sequences to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


def get_dataloaders(data_path: Path, batch_size: int = 32, split: str = "train") -> DataLoader:
    """
    Load the processed data from disk and return a PyTorch DataLoader for a specific split.

    Args:
        data_path (Path): Path to the processed dataset folder.
        batch_size (int): Number of samples per batch.
        split (str): The dataset split to load ("train", "validation", or "test").

    Returns:
        DataLoader: A PyTorch DataLoader for the requested split.
    """
    logger.info(f"Loading dataset from {data_path} for split '{split}'")
    dataset = load_from_disk(str(data_path))

    if split not in dataset:
        logger.error(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

    # Ensure the dataset is in PyTorch format
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Shuffle only for training
    shuffle = split == "train"

    data_loader = DataLoader(dataset[split], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    logger.info(f"DataLoader created: {len(data_loader)} batches, batch_size={batch_size}, shuffle={shuffle}")

    return data_loader


if __name__ == "__main__":
    app()
