Prima di lanciare qualsiasi cosa limitare la visibilità di GPU
export CUDA_VISIBLE_DEVICES="4"

Per creare il launcher:
python hierarchical_kmeans_launcher.py   --exp_dir ../data/custom   --embeddings_path ../features/features.npy   --config_file ../configs/curation/custom.yaml
sh data/custom/local_launcher.sh 

Environment utili:
base - general purpose
ssl-data-curation - python3.10 per data curation
thingsvision - per feature extraction
solo - per solo-learn

TODO:
snellire data pipeline. Informazioni non ripetute, features salvate in maniera compatta (HDF5?).
Strutturare meglio la folder tree