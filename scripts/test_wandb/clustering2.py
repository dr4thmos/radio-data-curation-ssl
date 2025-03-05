#!/usr/bin/env python3
import os
import wandb
import numpy as np
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clustering variant 2 toy script")
    parser.add_argument("--features_folder", type=str, default="./features",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_project",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="clustering_variant2",
        name="clustering_variant2_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "clustering", "variant": "variant2"}
    )
    
    # Recupera le features come fatto in clustering_variant1.py
    features_artifact = run.use_artifact("features:latest", type="dataset")
    features_dir = features_artifact.download()
    features_path = os.path.join(features_dir, "features.npy")
    features = np.load(features_path)
    
    # Simula un clustering diverso: calcola la mediana lungo l'asse 0
    cluster_result = np.median(features, axis=0)
    
    # Salva il risultato in un file JSON
    clustering_path = os.path.join(args.features_folder, "clustering_variant2.json")
    with open(clustering_path, "w") as f:
        json.dump({"cluster_result": cluster_result.tolist()}, f, indent=4)
    
    # Logga una metrica (ad esempio, la somma della mediana)
    wandb.log({"cluster_variant2_sum": float(cluster_result.sum())})
    
    # Logga l'artifact dei risultati del clustering variant2
    artifact = wandb.Artifact("clustering_variant2", type="results", description="Clustering variant 2 using median")
    artifact.add_file(clustering_path)
    run.log_artifact(artifact)
    
    run.finish()
    print("Clustering variant 2 completato.")

if __name__ == "__main__":
    main()
