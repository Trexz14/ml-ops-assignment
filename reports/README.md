# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [ ] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 

MLOps 63

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s234869, s245176, s244742, 

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used **Hugging Face Transformers** as our primary third-party framework. Specifically, we used `AutoTokenizer` for text tokenization with automatic padding, truncation, and encoding to convert raw text into model-ready tensors. We also used Hugging Face's `datasets` library to load the "SetFit/onestop_english" dataset directly from the Hugging Face Hub. For the model backbone, we used the pre-trained `prajjwal1/bert-mini` model, a lightweight BERT variant with only 11M parameters, making it ideal for rapid iteration during development. The framework saved us significant implementation time since we didn't need to write tokenization logic or attention mechanisms from scratch. We simply called `AutoTokenizer.from_pretrained(model_name)` and the framework handled all the preprocessing complexity. This allowed us to focus on building the MLOps infrastructure rather than spending time on NLP fundamentals.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used **uv** for dependency management because it's significantly faster than pip and keeps our builds deterministic. Everything we need is listed in `pyproject.toml`, while `uv` handles the `uv.lock` file to pin every single package version. This means the whole team is always running on the exact same setup, which pretty much killed any "it works on my machine" issues. To get the environment running, a new member just needs to install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), clone the repo, and run `uv sync` to set up the virtual environment and dependencies. Using `uv sync --frozen` ensures you're using the exact versions from the lock file without any surprises. Whenever we need a new tool, we just run `uv add <package>`, which keeps the `pyproject.toml` and lock file perfectly in sync and prevents dependency conflicts.



### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the cookiecutter template provided in the course, but the one with the agentcdocuments. We filled out the standard folders including `src/ml_ops_assignment/` for our source code (data.py, model.py, train.py, evaluate.py, api.py, visualize.py), `tests/` for pytest unit tests, `data/` for our processed dataset, `models/` for trained model checkpoints, `configs/` for configuration files, and `reports/` for this report. We also added `dockerfiles/` for our Docker containers and `docs/` for MkDocs documentation. The main deviation from the template was adding a `documents/` folder in the root directory to organize all agent-related documents (AGENTS.md, agent prompts) and project management files (checklist.md, project_description.md). This kept our root directory clean while maintaining easy access to documentation for both human developers and AI coding agents. We also use `tasks.py` with the `invoke` library instead of a Makefile for task automation, which provides better Python integration. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used Ruff for both linting and formatting since it's faster than traditional tools like flake8 and black. It enforces a 120-character line length and catches common Python issues. For type checking, we use mypy to verify type hints across our codebase. We tried to keep all functions and classes with consistent formatted docstrings explaining their purpose, parameters, and return values. For documentation, we created a QUICKSTART.md that provides quick setup instructions and common commands for running tests, training, evaluation, and API deployment. These practices matter in larger projects because they prevent bugs early, make code easier to understand for new team members, and ensure consistency across the codebase. Type hints catch type-related bugs at development time instead of runtime, while consistent formatting eliminates style debates and makes code reviews focus on logic. Documentation ensures that anyone can get started quickly without having to reverse-engineer the codebase.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 60 unit tests across three test files: test_data.py, test_model.py, and test_api.py. The data tests verify dataset loading from Hugging Face, batch collation, padding, and DataLoader functionality. Model tests check model initialization, forward pass correctness, output shapes, configuration loading, device handling, and the evaluation function. API tests validate the FastAPI health check endpoint, prediction endpoint functionality with various inputs including edge cases, and error handling for invalid requests. These tests ensure the critical components of our pipeline work correctly before deployment. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our total code coverage is 44%, which we know is far from 100%. However, we focused on testing all the main components we considered critical: data loading and preprocessing, model initialization and forward passes, and API endpoints. While higher coverage would be ideal, even 100% coverage wouldn't guarantee error-free code. Code coverage only measures which lines are executed during tests, not whether those tests check the right behavior or edge cases. You could have 100% coverage by simply running every line once without actually asserting anything meaningful. Real bugs often come from unexpected interactions between components, race conditions, incorrect business logic, or edge cases that weren't anticipated in the tests. Additionally, coverage doesn't catch issues like incorrect algorithm implementation where the code runs but produces wrong results. What matters more than coverage percentage is whether your tests validate critical functionality, catch common bugs, and test realistic scenarios and edge cases.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made extensive use of branches and pull requests throughout the project. Each team member created separate feature branches when working on specific tasks like implementing the API, setting up Docker containers, configuring CI/CD, or adding new model functionality. This approach prevented us from accidentally breaking the main branch with experimental or incomplete code. When a feature was complete and tested locally, we created a pull request to merge it into main. This gave other team members a chance to review the code, catch potential issues, and ensure everything integrated properly before merging. The PR workflow also helped us maintain a clean commit history and made it easy to track what changes were made for each feature. By keeping main stable and functional, we could always fall back to a working version if something went wrong on a feature branch. This branching strategy is essential for team collaboration and prevents the chaos of everyone pushing directly to main.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC to manage our processed datasets and trained models, storing them in Google Cloud Storage buckets. For our project, the main benefit was keeping large data files out of Git while still tracking them. Team members could clone the repo and simply run `uv run dvc pull` to download the data, which was way cleaner than sharing files manually or committing gigabytes to Git. Honestly, since our dataset was fairly static and we didn't do many preprocessing changes, DVC didn't make a huge difference for us day-to-day. However, DVC becomes really valuable in scenarios where you're frequently updating datasets, trying different preprocessing pipelines, or comparing model performance across different data versions. For example, if you discover data quality issues and need to rollback to a previous version, DVC lets you check out specific data versions just like Git commits. It's also crucial when running experiments with different data augmentation strategies or when your training data evolves over time and you need to track which model was trained on which dataset version.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We organized our CI into 4 separate GitHub Actions workflows in `.github/workflows/`. The **tests.yaml** workflow runs our 60 unit tests on every push and pull request to main. It uses a matrix strategy to test across three operating systems (Ubuntu, Windows, macOS) with Python 3.12, ensuring our code works consistently across platforms. The workflow uses `astral-sh/setup-uv@v7` with caching enabled (`enable-cache: true`) which significantly speeds up dependency installation by caching the uv cache directory between runs. On Ubuntu specifically, it authenticates with GCP via service account credentials stored in GitHub Secrets and pulls the processed data using DVC before running pytest with coverage reporting (`uv run coverage run -m pytest tests/`). The **linting.yaml** workflow enforces code quality by running Ruff for both linting (`ruff check .`) and formatting verification (`ruff format . --check`), plus mypy for static type checking to catch type errors before runtime. The **docker-build.yaml** workflow triggers automatically when changes are detected in `src/`, `dockerfiles/`, or `pyproject.toml`, authenticates with GCP, pulls DVC data and models, and submits the build to Google Cloud Build using `gcloud builds submit . --config cloudbuild.yaml`. This builds and pushes our Docker images to Artifact Registry. Finally, **pre-commit-update.yaml** runs daily to keep our pre-commit hooks updated with the latest versions. Our workflows can be seen here: https://github.com/Trexz14/ml-ops-assignment/actions

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We use YAML configuration files stored in `configs/experiments/`. The default config `default.yaml` contains all hyperparameters organized into sections: `model` (model_name, num_labels, dropout, hidden_size, max_length), `training` (optimizer, learning_rate, batch_size, num_epochs, gradient_clip, data_path), and `device` settings. To run training: `uv run python -m ml_ops_assignment.train --config-path configs/experiments/default.yaml`. The config is loaded via our `load_config()` function that parses the YAML into a dictionary. We also use Typer for CLI arguments, allowing checkpoint resumption: `uv run python -m ml_ops_assignment.train --checkpoint models/model_epoch_5.pt`.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We secured reproducibility through several mechanisms. First, all hyperparameters are stored in version-controlled YAML config files (`configs/experiments/default.yaml`), so every experiment is fully specified by its config. Second, we set `seed=42` in our data splitting code (`train_test_split(test_size=0.1, seed=42)`) to ensure consistent train/validation/test splits. Third, we use DVC to version our processed data and trained models, storing them in GCP buckets with `.dvc` files tracked in Git. This means anyone can reproduce our exact data state with `dvc pull`. Fourth, our `uv.lock` file pins all dependency versions exactly, eliminating "it works on my machine" issues. Fifth, we integrated Weights & Biases logging which automatically captures hyperparameters, metrics, and system info for each run. To reproduce an experiment: clone the repo, run `uv sync --frozen`, pull data with `dvc pull`, and run training with the same config file.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We developed three Docker images in `dockerfiles/`: **train.dockerfile** for model training, **api.dockerfile** for serving predictions via FastAPI, and **evaluate.dockerfile** for evaluation. All use the `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base image with uv for fast dependency installation. We use Docker's layer caching by copying `pyproject.toml` and `uv.lock` first, then running `uv sync --frozen --no-install-project` before copying source code. This means code changes don't invalidate the dependency cache. To build and run the API locally: `docker build -f dockerfiles/api.dockerfile -t api:latest .` then `docker run -p 8000:8000 api:latest`. Our `cloudbuild.yaml` automates building and pushing images to Google Artifact Registry when code changes are pushed to main. The images are tagged as `europe-west1-docker.pkg.dev/$PROJECT_ID/ml-ops-registry/api:latest` and `train:latest`. Link: [train.dockerfile](https://github.com/Trexz14/ml-ops-assignment/blob/main/dockerfiles/train.dockerfile)

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

For debugging, we primarily used our structured logging system built with **loguru**. Our `logging_config.py` sets up three log handlers: colored console output for immediate feedback, a rotating `app.log` file (10MB rotation, 7-day retention), and a separate `errors.log` for errors with full tracebacks (`backtrace=True, diagnose=True`). This made debugging production issues much easier since we could grep through logs for specific errors. For interactive debugging, we used VS Code's built-in Python debugger with breakpoints when stepping through complex logic like the model forward pass or data preprocessing pipeline. We also used print statements strategically during development, later converting them to proper logger calls. We did not perform formal profiling with tools like cProfile or PyTorch Profiler, though we did identify that tokenization was our main bottleneck and addressed it by using batched processing (`dataset.map(tokenize_function, batched=True)`).

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following GCP services: **Cloud Storage (Buckets)** for storing our processed datasets and trained model checkpoints, linked with DVC for version control (`gs://mlops-dtu-data/dvc-cache`). **Artifact Registry** for storing our Docker images (api:latest, train:latest) pushed via Cloud Build. **Cloud Build** for automated Docker image building, triggered by our GitHub Actions workflow which submits builds using `gcloud builds submit . --config cloudbuild.yaml`. **Cloud Run** for deploying our FastAPI inference service, which automatically scales based on incoming requests and only charges for actual compute time. We authenticated our CI/CD pipeline using a service account key stored in GitHub Secrets (`GCLOUD_SERVICE_KEY`).

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We implemented a FastAPI application in `src/ml_ops_assignment/api.py`. The API has two endpoints: a GET `/` health check returning `{"message": "OK", "status-code": 200}`, and a POST `/predict` endpoint for text classification. We used **Pydantic models** for request/response validation (`PredictRequest` with a `text` field, `PredictResponse` with `text`, `label`, `class_name`, and `status_code`). A key feature is our **lifespan context manager** that loads the model and tokenizer once at startup and stores them in a `model_assets` dictionary, avoiding repeated loading per request. The model predicts text difficulty on a 0-2 scale (Elementary, Intermediate, Advance). We handle the case where the model isn't available by returning mock predictions, which is useful for development and testing. The API includes proper logging at each step and uses `torch.no_grad()` during inference for efficiency. Run locally with: `uv run uvicorn ml_ops_assignment.api:app --reload`

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit testing we used **pytest** with FastAPI's `TestClient`. We have 4 test functions in `tests/test_api.py` that expand to 7 test cases: `test_read_root` verifies the health check endpoint returns 200 OK, `test_predict_endpoint_mock` tests the prediction endpoint returns valid responses with expected fields (text, label, class_name, status_code), and `test_predict_various_inputs` is a parametrized test with 4 different inputs including normal text, negative text, empty strings, and very long text to check edge cases. `test_predict_invalid_input` verifies that missing the required `text` field returns HTTP 422 (Unprocessable Entity). These tests run in our CI pipeline on every push. We did **not** implement formal load testing with tools like Locust or k6, though this would be valuable to determine how many concurrent requests our Cloud Run deployment can handle before latency degrades.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

fewafewubaofewnafioewnifowf ewafw afew afewafewafionewoanf waf ewonfieownaf fewnaiof newio fweanøf wea fewa
 fweafewa fewiagonwa ognwra'g
 wa
 gwreapig ipweroang w rag
 wa grwa
  g
  ew
  gwea g
  ew ag ioreabnguorwa bg̈́aw
   wa
   gew4igioera giroeahgi0wra gwa
