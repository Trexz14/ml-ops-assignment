import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ml_ops_assignment"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"PYTHONPATH=src uv run python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"PYTHONPATH=src uv run python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, checkpoint: str = "models/model_final.pt", split: str = "test") -> None:
    """Evaluate trained model on test set."""
    ctx.run(
        f"PYTHONPATH=src uv run python src/{PROJECT_NAME}/evaluate.py {checkpoint} {split}", echo=True, pty=not WINDOWS
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_evaluate(ctx: Context, checkpoint: str = "models/model_final.pt", split: str = "test") -> None:
    """Evaluate trained model using Docker (useful for Intel Mac users)."""
    # Ensure we have the evaluate Docker image
    result = ctx.run("docker images -q evaluate:latest", hide=True, warn=True)
    if not result or not result.stdout.strip():
        print("Docker image 'evaluate:latest' not found. Building it now...")
        docker_build(ctx)

    # Run evaluation in Docker container
    ctx.run(
        f'docker run --rm -v "$(pwd)/models":/app/models -v "$(pwd)/data":/app/data evaluate:latest {checkpoint} {split}',
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_train(ctx: Context, config: str = "configs/exp1.yaml") -> None:
    """Train model using Docker (useful for Intel Mac users)."""
    # Ensure we have the train Docker image
    result = ctx.run("docker images -q train:latest", hide=True, warn=True)
    if not result or not result.stdout.strip():
        print("Docker image 'train:latest' not found. Building it now...")
        docker_build(ctx)

    # Run training in Docker container
    ctx.run(
        "docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs train:latest",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def serve_api(ctx: Context) -> None:
    """Run the FastAPI server."""
    ctx.run(f"PYTHONPATH=src uv run uvicorn {PROJECT_NAME}.api:app --reload", echo=True, pty=not WINDOWS)
