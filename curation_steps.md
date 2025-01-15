```bash
git clone git@github.com:facebookresearch/ssl-data-curation.git
cd ssl-data-curation
conda create -n ssl-data-curation python=3.10
conda activate ssl-data-curation
pip install -r curation_requirements.txt
```


# Prepare the experiment
```bash
cd ssl-data-curation
mkdir -p data
cd scripts
python -c 'import numpy as np; np.save( "../data/100k_random.npy", np.random.randn(100000,256))'
python hierarchical_kmeans_launcher.py \
  --exp_dir ../data/2levels_random_embeddings \
  --embeddings_path ../data/100k_random.npy \
  --config_file ../configs/2levels_random_embeddings.yaml

cd ../data/2levels_random_embeddings
# Launch with slurm
bash launcher.sh
# Launch locally if only 1 node is used
# bash local_launcher.sh

cd ssl-data-curation/scripts
# Sampled indices will be saved in ssl-data-curation/data/2levels_random_embeddings/curated_datasets
PYTHONPATH=.. python run_hierarchical_sampling.py \
  --clustering_path ../data/2levels_random_embeddings \
  --target_size 20000 \
  --save
```