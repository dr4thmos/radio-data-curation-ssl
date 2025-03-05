#!/usr/bin/env python3
import wandb
import numpy as np
import os
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Features extraction toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="features",
        name="features_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "features"}
    )
    
    # Recupera l'artifact dei cutouts
    cutouts_artifact = run.use_artifact("cutouts:latest", type="dataset")
    cutouts_dir = cutouts_artifact.download()
    cutouts_path = os.path.join(cutouts_dir, "cutouts.json")
    with open(cutouts_path, "r") as f:
        cutouts = json.load(f)
    
    n = len(cutouts["cutouts"])
    # Simula l'estrazione di features: crea un array casuale di shape (n, 128)
    features = np.random.rand(n, 128)
    features_path = os.path.join(args.output_folder, "features.npy")
    np.save(features_path, features)
    
    # Logga l'artifact delle features come riferimento
    artifact = wandb.Artifact("features", type="dataset", description="Toy features estratte dai cutouts")
    abs_features_path = os.path.abspath(features_path)
    artifact.add_reference("file://" + abs_features_path)
    run.log_artifact(artifact)
    
    # Logga alcune metriche
    wandb.log({"features_mean": float(features.mean()), "features_std": float(features.std())})
    
    run.finish()
    print("Fase Features completata.")

if __name__ == "__main__":
    main()
