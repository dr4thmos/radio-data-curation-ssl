#!/usr/bin/env python3
import wandb
import numpy as np
import os
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clustering toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="clustering",
        name="clustering_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "clustering"}
    )
    
    # Recupera l'artifact delle features
    features_artifact = run.use_artifact("features:latest", type="dataset")
    features_dir = features_artifact.download()
    features_path = os.path.join(features_dir, "features.npy")
    features = np.load(features_path)
    
    n, d = features.shape
    # Simula il clustering: crea etichette casuali da 0 a 4
    labels = np.random.randint(0, 5, size=n)
    clustering_result = {"labels": labels.tolist()}
    
    clustering_path = os.path.join(args.output_folder, "clustering.json")
    with open(clustering_path, "w") as f:
        json.dump(clustering_result, f, indent=4)
    
    # Logga l'artifact del clustering come riferimento
    artifact = wandb.Artifact("clustering", type="results", description="Risultati toy del clustering")
    abs_clustering_path = os.path.abspath(clustering_path)
    artifact.add_reference("file://" + abs_clustering_path)
    run.log_artifact(artifact)
    
    wandb.log({"n_clusters": int(len(np.unique(labels)))})
    
    run.finish()
    print("Fase Clustering completata.")

if __name__ == "__main__":
    main()
