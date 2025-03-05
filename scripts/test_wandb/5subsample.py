#!/usr/bin/env python3
import wandb
import json
import os
import numpy as np
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Subsampling toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="subsampling",
        name="subsampling_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "subsampling"}
    )
    
    # Recupera l'artifact del clustering
    clustering_artifact = run.use_artifact("clustering:latest", type="results")
    clustering_dir = clustering_artifact.download()
    clustering_path = os.path.join(clustering_dir, "clustering.json")
    with open(clustering_path, "r") as f:
        clustering_result = json.load(f)
    
    labels = np.array(clustering_result["labels"])
    n = len(labels)
    # Seleziona un sotto-campione: ad esempio, il 20% degli indici
    indices = np.random.choice(n, size=max(1, n // 5), replace=False)
    subsample = {"indices": indices.tolist()}
    
    subsample_path = os.path.join(args.output_folder, "subsample.json")
    with open(subsample_path, "w") as f:
        json.dump(subsample, f, indent=4)
    
    artifact = wandb.Artifact("subsample", type="results", description="Sottocampione dei risultati del clustering")
    abs_subsample_path = os.path.abspath(subsample_path)
    artifact.add_reference("file://" + abs_subsample_path)
    run.log_artifact(artifact)
    
    wandb.log({"subsample_size": len(indices)})
    
    run.finish()
    print("Fase Subsampling completata.")

if __name__ == "__main__":
    main()
