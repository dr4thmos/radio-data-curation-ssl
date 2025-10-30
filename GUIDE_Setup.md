# GUIDE: Environment Setup

This document provides instructions for configuring the necessary Python environments to run the data curation pipeline. Adherence to these steps is required for correct operation.

## Target Environment Context

This pipeline was developed and tested on the Leonardo HPC system at CINECA. The setup instructions reflect this context, using `venv` for environment management, which is compatible with job schedulers like Slurm. For detailed information on the Leonardo system, refer to the official [CINECA HPC documentation](https://docs.hpc.cineca.it/index.html).

## The Two-Environment Requirement

The pipeline requires two separate Python virtual environments due to conflicting package dependencies between different stages.

*   **`feat_extract_env` (Python 3.9):** Used for data preparation (`cutouts.py`, `merge.py`) and feature extraction (`feature_extraction.py`).
*   **`clustering_env` (Python 3.10):** Used for unsupervised clustering (`clustering.py`) and sampling (`scripts/random_sampling.py`).

### Prerequisites

Before proceeding, ensure the following are available in your shell environment:
*   `git`
*   A Python 3.9 interpreter (e.g., `python3.9`)
*   A Python 3.10 interpreter (e.g., `python3.10`)

### Step-by-Step Environment Creation

Execute the following steps from your terminal.

#### 1. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone https://github.com/dr4thmos/radio-data-curation-ssl.git
cd radio-data-curation-ssl
```

#### 2. Create the Feature Extraction Environment (feat_extract_env)
This environment handles the initial data processing and feature extraction stages. 

```bash
# Create the virtual environment using your Python 3.9 interpreter
python3.9 -m venv feat_extract_env

# Activate the environment
source feat_extract_env/bin/activate

# Install the required packages from the specified requirements file
pip install -r requirements/requirements_feat_extract.txt

# Deactivate the environment once installation is complete
deactivate
```

#### 3. Create the Clustering Environment (clustering_env)
This environment handles the final curation stages, which have different dependencies.

```bash
# Create the virtual environment using your Python 3.10 interpreter
python3.10 -m venv clustering_env

# Activate the environment
source clustering_env/bin/activate

# Install the required packages from the specified requirements file
pip install -r requirements/requirements_clustering.txt

# Deactivate the environment once installation is complete
deactivate
```

### Environment Activation
Throughout the pipeline's execution, you will need to activate the appropriate environment for each script. This is done using the source command as shown above. The Pipeline Usage Guide will specify which environment is required for each step.
