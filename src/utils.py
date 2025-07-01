# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from pathlib import Path

import numpy as np
import torch

import json
import hashlib
import mlflow
import re


def format_readable_run_name(base_name, params_key_value, hash):
    readable_run_name = base_name
    for key, value in params_key_value.items(): # da fare per i dict
        readable_run_name = readable_run_name + "-" + sanitize(str(key)) + "_" + sanitize(str(value))
    return readable_run_name + "_" + hash

def compute_config_hash(config: dict, exclude_keys=None) -> str:
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)

    filtered_config = {k: v for k, v in config.items() if k not in exclude_keys}
    config_str = json.dumps(filtered_config, sort_keys=True)
    
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]
"""
def compute_config_hash(config: dict) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]
"""
def log_on_mlflow(mlflow_experiment_name, mlflow_run_name, metadata, metadata_file):

    try:
        mlflow.create_experiment(mlflow_experiment_name)
    except:
        print("Probabilmente l'esperimento mlflow è già presente")

    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        print(f"Active run_id: {run.info.run_id}")
        metadata["mlflow_run_id"] = run.info.run_id
        #mlflow.note.content("Optional run description")
        #run = mlflow.active_run()
        
        for key, value in metadata.items():
            mlflow.log_param(key, value)
        
        if metadata_file is not None:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

            mlflow.log_artifact(metadata_file, artifact_path="")



def sanitize(filename: str, replacement: str = "") -> str:
    """
    Sanifica un nome di file rimuovendo caratteri non validi e sostituendo i punti.
    
    Args:
        filename (str): Il nome del file senza estensione da sanificare.
        replacement (str): Il carattere con cui sostituire i caratteri non validi.
    
    Returns:
        str: Il nome sanificato del file.
    """
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F.]'  # Include il punto nei caratteri vietati
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Rimuove eventuali spazi o caratteri di sostituzione in eccesso ai bordi
    sanitized = sanitized.strip(replacement)

    return sanitized[:255]  # Assicura che non superi i limiti di filesystem


def create_clusters_from_cluster_assignment(
    cluster_assignment: np.array,
    num_clusters: int,
    return_object_array: bool = True,
):
    """
    Build clusters from cluster assignment.
    """
    ID = np.argsort(cluster_assignment)
    sorted_cluster_assigment = cluster_assignment[ID]
    index_split = np.searchsorted(sorted_cluster_assigment, list(range(num_clusters)))
    clusters = np.split(ID, index_split[1:])
    if return_object_array:
        return np.array(clusters, dtype=object)
    else:
        return clusters


def find_all_checkpoints(save_dir, pattern):
    """
    Parameters:
        pattern: str
            checkpoint name format <filename>_%d.<file extension>,
            e.g., kmpp_checkpoint_%d.pth
    """
    save_dir = Path(save_dir)
    ckpt_list = [str(el.stem) for el in save_dir.glob(pattern.replace("%d", "*"))]
    ckpt_list = [int(el.split("_")[-1]) for el in ckpt_list]
    ckpt_list = sorted(ckpt_list)
    return [Path(save_dir, pattern % el) for el in ckpt_list]


def get_last_valid_checkpoint(save_dir, pattern):
    """
    Find path to the last checkpoint.
    """
    ckpt_list = find_all_checkpoints(save_dir, pattern)
    for ckpt_path in ckpt_list[::-1]:
        try:
            if ".pth" in pattern:
                _ = torch.load(ckpt_path, map_location="cpu")
            elif ".npy" in pattern:
                _ = np.load(ckpt_path)
            else:
                raise ValueError("Pattern not recognized!")
            return ckpt_path
        except Exception:
            continue
    return None


def _delete_old_checkpoint(
    save_dir, current_iter, checkpoint_period, max_num_checkpoints, pattern
):
    Path(
        save_dir, pattern % (current_iter - checkpoint_period * max_num_checkpoints)
    ).unlink(missing_ok=True)


def setup_logging(
    *,
    name: str = None,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> None:
    """
    Basic setting for logger.
    """
    logging.captureWarnings(capture_warnings)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return

    fmt_prefix = (
        "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    )
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.propagate = False
    logger.addHandler(handler)
    return

