# GUIDE: End-to-End Pipeline Usage

This document provides a tutorial-style walkthrough of the data curation pipeline, with runnable commands for each stage.


## Managing Pipeline Steps: Finding Run IDs

This pipeline uses MLflow to log all experiments, parameters, and artifacts. Every execution of a pipeline script is recorded as a "run." To chain steps together, you must provide the unique **Run ID** of a previous step as input to the next.

There are two primary methods for retrieving a Run ID:

### Method 1: The MLflow UI (Recommended for Inspection)

The MLflow User Interface provides a powerful web-based dashboard to visually explore your runs, compare parameters, and view artifacts. This is the best way to inspect the pipeline's progress.

To launch the UI, run the following command from the project's root directory:

```bash
# This command starts a local web server to view the MLflow dashboard
mlflow ui
```
Navigate to the provided URL (usually http://127.0.0.1:5000). In the UI, you can click on an experiment and then a specific run to see its details. The Run ID is a long alphanumeric string that you can copy directly from this view.



Method 2: The find_run.py Script (Recommended for Automation)
For convenience in scripting, this project includes the scripts/find_run.py utility. It programmatically finds the Run ID of the most recent execution of a given script, which is ideal for creating automated workflows.

```bash
# This example finds the ID of the last feature_extraction run and stores it
FEATURE_RUN_ID=$(python scripts/find_run.py --run-name feature_extraction)
```

#### Usage in this Guide
For the remainder of this guide, the examples will use the find_run.py method to allow for a fully automated, copy-pasteable workflow. However, you can always replace the $(python ...) part of a command with a Run ID that you have manually copied from the MLflow UI.

## End-to-End Pipeline Walkthrough

### Step 1: Create Cutouts (`cutouts.py`)

The first step is to ingest raw FITS mosaic files and generate smaller, uniform images known as cutouts. This script processes a directory of mosaics and saves the resulting cutouts to a structured output directory.

**Environment:** `feat_extract_env`

**Activate:** `source feat_extract_env/bin/activate`

**Command:**
```bash
python scripts/cutouts.py \
    --mosaics_path "/path/to/your/mosaics" \
    --output_path "/path/to/your/outputs" \
    --window_size 128
```

## Step 2: Merge Cutout Lists (merge.py)
This script aggregates the metadata from multiple generated cutouts into a single, unified list. This list serves as the manifest for all subsequent processing steps.

**Environment:** feat_extract_env
**Activate:** source feat_extract_env/bin/activate

**Command:**
```bash
# Find the run ID for the 64x64 cutout run by filtering on the parameter
CUTOUTS_RUN_ID_64=$(python scripts/find_run.py --run-name cutting_out --window_size 64)

# Find the run ID for the 128x128 cutout run
CUTOUTS_RUN_ID_128=$(python scripts/find_run.py --run-name cutting_out --window_size 128)

# Run the merge script, providing both run IDs as a space-separated list
python scripts/merge.py \
    --cutouts-id "$CUTOUTS_RUN_ID_64" "$CUTOUTS_RUN_ID_128"
```
#### Step 3: Extract Features (feature_extraction.py)
This script uses a deep learning model to process each cutout image and compute a corresponding feature vector. These vectors are numerical representations of the image content, which are essential for the clustering stage.

**Environment:** feat_extract_env

**Activate:** source feat_extract_env/bin/activate

**Command:**
```bash
# Find the run ID from the previous merge step
MERGED_RUN_ID=$(python scripts/find_run.py --run-name merge_cutouts --cutout-ids "[CUTOUTS_RUN_ID_64, CUTOUTS_RUN_ID_128]")

python scripts/feature_extraction.py \
    --merged-cutouts-id "$MERGED_RUN_ID"
```

## Step 4: Unsupervised Clustering (Two-Part Process)
The clustering stage is a two-part process. The first command prepares the experiment and generates a launch script. The second command executes this script to run the computationally intensive job.

**Environment:** clustering_env

**Activate:** source clustering_env/bin/activate

#### Step 4a: Prepare the Clustering Experiment
This script configures the clustering algorithm (e.g., K-Means). It takes the feature vectors as input and creates a new experiment directory containing a run.sh script.
**Command:**
```bash
# Find the run ID from the previous feature extraction step
FEATURE_RUN_ID=$(python scripts/find_run.py --run-name feature_extraction)

python scripts/4_clustering.py \
    --features-id "$FEATURE_RUN_ID" \
    --config_file "configs/clusters/A_3M_3lvl_60k_4k_300.yaml" \
    --slurm_account "your_account" \
    --slurm_partition "your_partition" \
    --slurm_time "01:00:00"
```
#### Step 4b: Execute the Clustering Job
Next, you must navigate into the directory created above and execute the generated script.
**Command:**
```bash
# Find the path to the clustering experiment directory
CLUSTERING_DIR_PATH=$(python scripts/find_run.py --run-name clustering --get-path)

echo "Changing directory to: $CLUSTERING_DIR_PATH"
cd "$CLUSTERING_DIR_PATH"

# Execute the generated shell script to run the actual clustering
./run.sh

# Return to the project root directory to continue
cd -
```
#### Step 5: Curated Sampling (random_sampling.py)
The final step is to perform sampling based on the clustering results. This script uses the cluster assignments to select a diverse, representative subset of the data, producing the final curated dataset.

**Environment:** clustering_env

**Activate:** source clustering_env/bin/activate

**Command:**

```bash
# Find the run ID from the clustering job
CLUSTERING_RUN_ID=$(python scripts/find_run.py --run-name clustering)

python scripts/random_sampling.py \
    --clustering-id "$CLUSTERING_RUN_ID" \
    --num-samples 10000
```