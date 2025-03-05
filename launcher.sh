#!/bin/bash
#bash launch_kmeans.sh [CUDA_VISIBLE_DEVICES] [EXP_NAME]

# Valori di default
CUDA_VISIBLE_DEVICES_DEFAULT="4"
EXP_NAME_DEFAULT="test"

# Leggi argomenti dalla linea di comando o usa i valori di default
CUDA_VISIBLE_DEVICES=${1:-$CUDA_VISIBLE_DEVICES_DEFAULT}
EXP_NAME=${2:-$EXP_NAME_DEFAULT}

# Esporta le variabili
export CUDA_VISIBLE_DEVICES
export EXP_NAME

# Esegui lo script Python
python scripts/hierarchical_kmeans_launcher.py \
    --exp_dir data/$EXP_NAME \
    --embeddings_path features/features.npy \
    --config_file configs/curation/$EXP_NAME.yaml \
    --scripts_path $PWD/scripts/

# Esegui il launcher locale
sh data/$EXP_NAME/local_launcher.sh
