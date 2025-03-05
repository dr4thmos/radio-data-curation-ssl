#!/usr/bin/env python3
import wandb
import json
import os
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Dataset creation toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Genera un ID univoco per l'esperimento
    experiment_id = "toy_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inizializza W&B per la fase dataset
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",  # Sostituisci con il tuo username o team
        group=experiment_id,
        job_type="dataset",
        name="dataset_" + experiment_id,
        config={"phase": "dataset"}
    )
    
    # Crea un dataset fittizio: ad esempio una lista di nomi di immagini
    dataset = {"images": [f"image_{i}.jpg" for i in range(50)]}
    dataset_path = os.path.join(args.output_folder, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=4)
    
    # Logga l'artifact come riferimento (senza caricare il file)
    artifact = wandb.Artifact("dataset", type="dataset", description="Toy dataset")
    abs_dataset_path = os.path.abspath(dataset_path)
    artifact.add_reference("file://" + abs_dataset_path)
    run.log_artifact(artifact)
    
    run.finish()
    print(f"Dataset creato. Usa questo EXPERIMENT_ID per le fasi successive: {experiment_id}")

if __name__ == "__main__":
    main()
