# MLOps Project Task Description

## Overview
As a newly hired MLOps engineer at a start-up, your primary objective is to develop a complete MLOps pipeline for a specific task. Unlike traditional machine learning roles, your success is measured by the speed and efficiency of setting up the pipeline rather than the absolute performance of the model.

## Project Objectives
The project accounts for approximately **1/3 of the course time** and serves as the basis for your exam. The core goals include:

- **Collaboration**: Working effectively in a group of 3–5 students.
- **Formulation**: Defining a project within provided guidelines.
- **Application**: Implementing course materials (version control, CI/CD, deployment) to a real-world problem.
- **Presentation**: Documenting and presenting findings through a technical report.

## Project Lifecycle & Strategy
The project follows a standard MLOps loop, but you are encouraged to **fast-track** the *Design* and *Model Development* phases to focus on *Operations*.

### 1. Selection Strategy

#### Data
- Choose a dataset that is not overly complex.
- Ideally **< 1GB**, and definitely **< 10GB**.
- Avoid datasets with many small files, as they slow down version control tools like DVC.

#### Model
- Find a **sweet spot**: harder than basic benchmarks like MNIST, but easier than training a Large Language Model (LLM) from scratch.

#### Open-Source Leverage
- Use existing frameworks to get **80% of the way**.
- **PyTorch** is the required base framework.
- Recommended libraries include:
  - **Hugging Face Transformers**: For NLP tasks.
  - **MONAI**: For healthcare imaging.
  - **PyTorch Geometric**: For graph-based data.

### 2. Implementation Checklist
The project is divided into three weekly phases. You do **not** need to complete every item to pass, as the list is exhaustive.

| Phase  | Key Tasks |
|------|----------|
| **Week 1** | Create Git repo, set up Conda environment, use Cookiecutter for structure, implement DVC for data versioning, and use Docker for containerization. |
| **Week 2** | Write unit tests, implement Continuous Integration (CI) on GitHub, set up GCP storage (Buckets), and build a FastAPI for inference. |
| **Week 3** | Implement drift detection, set up cloud monitoring/alerts in GCP, and optimize performance through quantization or distributed training. |

## Evaluation Criteria
Evaluation is based on the technical quality of the pipeline and the depth of collaboration. Specific focus areas include:

- **Reproducibility**: How well code, data, and experiments are version-controlled.
- **Automation**: Proper implementation of Continuous Integration (CI).
- **Deployment**: A final model must be deployed online and interactable for end-users.
- **Documentation**: Clear documentation of essential code parts and hyperparameters.

## Hand-in Requirements
The final submission consists of a **Project Report in Markdown format**, which will be automatically scraped from your repository on **January 23rd at 23:59**.

### Required Repository Structure
```txt
project_repo/
├── data/                # Raw and processed data folders
├── models/              # Model definitions
├── reports/
│   └── README.md        # Your Final Report
├── figures/             # Required architectural diagrams
└── src/                 # Source code
```