import os
import re

# Percorso della directory principale
base_dir = "cutouts"

# Pattern per identificare i file info<altro>.json
pattern = re.compile(r"info.+\.json")

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if pattern.match(file):
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, "info.json")
            try:
                os.rename(old_path, new_path)
                print(f"Rinominato: {old_path} -> {new_path}")
            except FileExistsError:
                print(f"Impossibile rinominare {old_path} perché esiste già un file con nome info.json in {root}.")
