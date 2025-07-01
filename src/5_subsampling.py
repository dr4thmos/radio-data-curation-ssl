from datetime import datetime
import os
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow
from parsers.subsampling_parser import get_parser
import mlflow
import json
import numpy as np
import hierarchical_sampling as hs
from clusters import HierarchicalCluster

MLFLOW_EXPERIMENT_NAME = "subsampling"
MLFLOW_RUN_NAME = "subsampling"
SUBSAMPLED_INDICES_BASE_FILENAME = "subsampled_indices"

args = get_parser()
metadata=vars(args)

metadata["timestamp"] = datetime.now().isoformat()
config_hash = compute_config_hash(metadata)
metadata["config_hash"] = config_hash

readable_name_params = {}
readable_name_params["target_size"] = metadata.get("target_size")

subsampled_indices_filename = format_readable_run_name(SUBSAMPLED_INDICES_BASE_FILENAME, readable_name_params, config_hash)
subsampled_indices_filename = subsampled_indices_filename + ".npy"
metadata["subsampled_indices_filename"] = subsampled_indices_filename

try:
    run = mlflow.get_run(metadata["clusters_id"])
    cluster_path = run.data.params.get("exp_dir")
    cluster_levels = run.data.params.get("n_levels")
    metadata["exp_dir"] = cluster_path
    
except mlflow.exceptions.MlflowException as e:
    print(f"Errore nel recuperare il cutout_id {metadata['clusters_id']}: {e}")

cl = HierarchicalCluster.from_file(cluster_path, levels=int(cluster_levels))
sampled_indices = hs.hierarchical_sampling(cl, target_size=metadata["target_size"])
np.save(os.path.join(cluster_path, metadata["subsampled_indices_filename"]), sampled_indices)

# create a numpy array with metadata["target_size"] rows, 1 column filled with random indices non duplicate in a range from 0 to metadata["target_size"]-1
#sampled_indices = np.random.choice(metadata["target_size"], metadata["target_size"], replace=False)
#np.save(os.path.join(run_path, metadata["subsampled_indices_filename"]), sampled_indices)


log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, metadata, None)