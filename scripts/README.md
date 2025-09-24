umap_vis_paper.py
serve per creare plot umap delle 3M features + sovrapposte le features con labels
si lancia con slurm, ogni modifica agli arguments deve rispecchiare una modifica in slurm
Sono stati tolti i riferimenti a mlflow per questa fase di analisi, mlflow è utile quando si ha una pipeline stabile, altimenti si sovraingegnerizza limitando la flessibilità.

sbatch slurms/umap_viz     --unlabeled_features_path "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__64effbed/features.npy"     --labeled_features_path "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__b27e541a/features_data_fast.h5"     --labeled_info_json_path "/leonardo_work/INA24_C5B09/lofar_rgz_dataset/lotss_dr2_morph_horton-dataset/data/info.json"     --output_dir "outputs/umap" --min_dist 0.1 --n_neighbors 15 --metric euclidean --pca_components 64