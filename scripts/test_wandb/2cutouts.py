#!/usr/bin/env python3
import wandb
import json
import os
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cutouts toy script")
    parser.add_argument("--output_folder", type=str, default="./data",
                        help="Cartella in cui salvare i file generati")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Recupera l'EXPERIMENT_ID dalla variabile d'ambiente
    experiment_id = os.environ.get("EXPERIMENT_ID", "toy_experiment_default")
    
    run = wandb.init(
        project="toy_pipeline",
        entity="ssl-inaf",
        group=experiment_id,
        job_type="cutouts",
        name="cutouts_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={"phase": "cutouts"}
    )
    
    # Recupera l'artifact del dataset
    dataset_artifact = run.use_artifact("dataset:latest", type="dataset")
    dataset_dir = dataset_artifact.download()  # Scarica in una cartella temporanea
    dataset_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Simula l'estrazione di cutouts: ad es. sostituisci "image" con "cutout"
    cutouts = {"cutouts": [img.replace("image", "cutout") for img in dataset["images"]]}
    cutouts_path = os.path.join(args.output_folder, "cutouts.json")
    with open(cutouts_path, "w") as f:
        json.dump(cutouts, f, indent=4)
    
    # Logga l'artifact dei cutouts come riferimento
    artifact = wandb.Artifact("cutouts", type="dataset", description="Toy cutouts estratti dal dataset")
    abs_cutouts_path = os.path.abspath(cutouts_path)
    artifact.add_reference("file://" + abs_cutouts_path)
    run.log_artifact(artifact)
    
    run.finish()
    print("Fase Cutouts completata.")

if __name__ == "__main__":
    main()
