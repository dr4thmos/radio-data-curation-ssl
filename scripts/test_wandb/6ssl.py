#!/usr/bin/env python3
import wandb
import os
import json
import numpy as np
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Self-supervision toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="self_supervision",
        name="self_supervision_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "self_supervision"}
    )
    
    # Recupera l'artifact del subsampling
    subsample_artifact = run.use_artifact("subsample:latest", type="results")
    subsample_dir = subsample_artifact.download()
    subsample_path = os.path.join(subsample_dir, "subsample.json")
    with open(subsample_path, "r") as f:
        subsample = json.load(f)
    
    # Simula il training self-supervision: crea un "modello" fittizio (parametri casuali)
    model = {"weights": np.random.rand(10).tolist(), "bias": np.random.rand(1).tolist()}
    model_path = os.path.join(args.output_folder, "model.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=4)
    
    artifact = wandb.Artifact("self_supervision_model", type="model", description="Modello toy self-supervision")
    abs_model_path = os.path.abspath(model_path)
    artifact.add_reference("file://" + abs_model_path)
    run.log_artifact(artifact)
    
    wandb.log({"model_weight_sum": float(np.sum(model["weights"]))})
    
    run.finish()
    print("Fase Self Supervision completata.")

if __name__ == "__main__":
    main()
