# export EXP_NAME="try_1"
# export FEATURES_NAME="clip"
# export CUDA_VISIBLE_DEVICES="4"
# python scripts/hierarchical_kmeans_launcher.py   --exp_dir data/$EXP_NAME   --embeddings_path features/features_clip/features.npy   --config_file configs/curation/$EXP_NAME.yaml  --scripts_path $PWD/scripts/
# python scripts/hierarchical_kmeans_launcher.py   --exp_dir data/${EXP_NAME}_${FEATURES_NAME}   --embeddings_path features/features_${FEATURES_NAME}/features.npy   --config_file configs/curation/$EXP_NAME.yaml  --scripts_path $PWD/scripts/
# aggiungere alla cartella di data curation un file json su che feature sono state usate