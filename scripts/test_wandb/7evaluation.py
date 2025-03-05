#!/usr/bin/env python3
import wandb
import os
import json
import numpy as np
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluation toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="evaluation",
        name="evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "evaluation"}
    )
    
    # Recupera l'artifact del modello self-supervision
    model_artifact = run.use_artifact("self_supervision_model:latest", type="model")
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, "model.json")
    with open(model_path, "r") as f:
        model = json.load(f)
    
    # Simula una valutazione: calcola una accuracy fittizia basata sui parametri del modello
    weight_sum = np.sum(model["weights"])
    accuracy = 0.5 + (weight_sum % 0.5)  # formula fittizia
    evaluation = {"accuracy": accuracy}
    
    evaluation_path = os.path.join(args.output_folder, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=4)
    
    artifact = wandb.Artifact("evaluation", type="results", description="Risultati toy della valutazione")
    artifact.add_file(evaluation_path)
    run.log_artifact(artifact)
    
    wandb.log({"accuracy": accuracy})
    
    run.finish()
    print("Fase Evaluation completata.")

if __name__ == "__main__":
    main()
