Generare lista immagini random:

`python scripts/random_sampling.py --input /leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/merged_cutou
ts/merged_cutouts_74003e44/info.json --output outputs/rand_sample1_30k.npy --num_samples 30000`

Runnare esperiento su immagini random:
- Creare file di configurazione esempio solo-learn-radio/scripts/pretrain/data_curation_pipeline/SR1
    - Sostituire subset_indices_path e modificare il campo name
- Creare slurm code (copiare da template e sostituire con nome della configurazione)
    - `python main_pretrain_custom.py --config-path /leonardo_work/INA24_C5B09/solo-learn-radio/scripts/pretrain/data_curation_pipeline/ --config-name SR3.yaml`

Syncronizzare i run wandb:
- `wandb sync solo-learn-radio/wandb/offline-run-20250916_171733-ju7q7ea8`

Visualizzazione:
Indicare, oltre agli iperparametri, i file contenenti le features unlabeled, quelle labeled e il file con le informazioni relative alle etichette:
```sh
sbatch slurms/umap_viz     --unlabeled_features_path "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__64effbed/features.npy"     --labeled_features_path "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__b27e541a/features_data_fast.h5"     --labeled_info_json_path "/leonardo_work/INA24_C5B09/lofar_rgz_dataset/lotss_dr2_morph_horton-dataset/data/info.json"     --output_dir "outputs/umap/dinov2" --min_dist 0.0 --n_neighbors 125 --metric euclidean --pca_components 32 --marker_radius 0.1 --marker_linewidth 0.0
```


