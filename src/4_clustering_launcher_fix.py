#### code for fixing missing MLFLOW runs

from argparse import ArgumentParser
from pathlib import Path
import os
from omegaconf import OmegaConf

import mlflow
from datetime import datetime
from utils import format_readable_run_name, compute_config_hash, log_on_mlflow


ROOT = Path().resolve()
MEMORY_LIMIT = 2e8

def write_main_script(cfg, level_dir, level_id, scripts_path):
    """
    Write slurm script for a level of k-means.
    """
    save_dir = level_dir
    if cfg.n_splits[level_id - 1] > 1:
        save_dir = Path(level_dir, "pre_clusters")
        Path(save_dir, "logs").mkdir(parents=True)

    chunk_size = int(MEMORY_LIMIT / cfg.n_clusters[level_id - 1])
    if level_id == 1:
        data_path = Path(cfg.embeddings_path).resolve()
    else:
        data_path = Path(cfg.exp_dir, f"level{level_id-1}/centroids.npy").resolve()

    with open(Path(level_dir, "slurm_script.s"), "w") as f:
        f.write(
f"""#!/usr/bin/env bash

#SBATCH --requeue
#SBATCH --nodes={cfg.nnodes[level_id-1]}
#SBATCH --gpus-per-node={cfg.ngpus_per_node[level_id-1]}
#SBATCH --ntasks-per-node={cfg.ngpus_per_node[level_id-1]}
#SBATCH --job-name=kmeans_level{level_id}
#SBATCH --output={save_dir}/logs/%j_0_log.out
#SBATCH --error={save_dir}/logs/%j_0_log.err
#SBATCH --signal=USR2@300
#SBATCH --open-mode=append\n"""
        )
        if cfg.ncpus_per_gpu is not None:
            f.write(f"#SBATCH --cpus-per-task={cfg.ncpus_per_gpu}\n")
        if cfg.slurm_partition is not None:
            f.write(f"#SBATCH --partition={cfg.slurm_partition}\n")
        if cfg.slurm_qos is not None:
            f.write(f"#SBATCH --qos={cfg.slurm_qos}\n")
        if cfg.slurm_account is not None:
            f.write(f"#SBATCH --account={cfg.slurm_account}\n")
        if cfg.slurm_time is not None:
            f.write(f"#SBATCH --time={cfg.slurm_time}\n")


        f.write(f"""
module load cuda cudnn nccl openmpi spack/0.21.0-68a
spack load python@3.10
source /leonardo_work/INA24_C5B09/clustering/bin/activate
EXPDIR={save_dir}
cd {scripts_path}

PYTHONPATH=.. \\
srun --unbuffered --output="$EXPDIR"/logs/%j_%t_log.out --error="$EXPDIR"/logs/%j_%t_log.err \\
    python -u run_distributed_kmeans.py \\
    --data_path {data_path} \\
    --n_clusters {cfg.n_clusters[level_id-1]} \\
    --n_iters {cfg.n_iters} \\
    --chunk_size {chunk_size} \\
    --dtype {cfg.dtype} \\
    --high_precision {cfg.high_precision} \\
    --checkpoint_period {cfg.checkpoint_period} \\
    --exp_dir $EXPDIR \\
    --n_steps {cfg.n_resampling_steps[level_id-1]} \\
    --sample_size {cfg.sample_size[level_id-1]} \\
    --sampling_strategy {cfg.sampling_strategy}"""
        )
        if level_id == 1 and cfg.subset_indices_path is not None:
            f.write(f" \\\n    --subset_indices_path {cfg.subset_indices_path}\n")
        else:
            f.write("\n")

    with open(Path(level_dir, "local_script.sh"), "w") as f:
        f.write(
f"""#!/usr/bin/env bash
EXPDIR={save_dir}
cd {scripts_path}

PYTHONPATH=.. \\
torchrun \\
--nnodes={cfg.nnodes[level_id-1]} \\
--nproc_per_node={cfg.ngpus_per_node[level_id-1]} \\
    run_distributed_kmeans.py \\
    --use_torchrun \\
    --data_path {data_path} \\
    --n_clusters {cfg.n_clusters[level_id-1]} \\
    --n_iters {cfg.n_iters} \\
    --chunk_size {chunk_size} \\
    --dtype {cfg.dtype} \\
    --high_precision {cfg.high_precision} \\
    --checkpoint_period {cfg.checkpoint_period} \\
    --exp_dir $EXPDIR \\
    --n_steps {cfg.n_resampling_steps[level_id-1]} \\
    --sample_size {cfg.sample_size[level_id-1]} \\
    --sampling_strategy {cfg.sampling_strategy}"""
        )
        if level_id == 1 and cfg.subset_indices_path is not None:
            f.write(f" \\\n    --subset_indices_path {cfg.subset_indices_path}\n")
        else:
            f.write("\n")


def write_split_clusters_script(cfg, level_dir, level_id):
    """
    Write slurm script to split pre-clusters into smaller ones if necessary.
    """
    if level_id == 1:
        data_path = Path(cfg.embeddings_path).resolve()
    else:
        data_path = Path(cfg.exp_dir, f"level{level_id-1}/centroids.npy").resolve()

    with open(Path(level_dir, "slurm_split_clusters_script.s"), "w") as f:
        f.write(
f"""#!/usr/bin/env bash

#SBATCH --requeue
#SBATCH --nodes={cfg.nnodes[level_id-1]}
#SBATCH --gpus-per-node={cfg.ngpus_per_node[level_id-1]}
#SBATCH --ntasks-per-node={cfg.ngpus_per_node[level_id-1]}
#SBATCH --job-name=split_kmeans_level{level_id}
#SBATCH --output={level_dir}/logs/%j_0_log.out
#SBATCH --error={level_dir}/logs/%j_0_log.err
#SBATCH --signal=USR2@300
#SBATCH --open-mode=append\n"""
        )
        if cfg.ncpus_per_gpu is not None:
            f.write(f"#SBATCH --cpus-per-task={cfg.ncpus_per_gpu}\n")
        if cfg.slurm_partition is not None:
            f.write(f"#SBATCH --partition={cfg.slurm_partition}\n")
        if cfg.slurm_qos is not None:
            f.write(f"#SBATCH --qos={cfg.slurm_qos}\n")
        if cfg.slurm_account is not None:
            f.write(f"#SBATCH --account={cfg.slurm_account}\n")
        if cfg.slurm_time is not None:
            f.write(f"#SBATCH --time={cfg.slurm_time}\n")

        f.write(f"""
module load cuda cudnn nccl openmpi spack/0.21.0-68a
spack load python@3.10
source /leonardo_work/INA24_C5B09/clustering/bin/activate
EXPDIR={level_dir}
cd {scripts_path}

PYTHONPATH=.. \\
srun --unbuffered --output="$EXPDIR"/logs/%j_%t_log.out --error="$EXPDIR"/logs/%j_%t_log.err \\
    python -u split_clusters.py \\
    --data_path {data_path} \\
    --clusters_path "$EXPDIR"/pre_clusters/sorted_clusters.npy \\
    --n_splits {cfg.n_splits[level_id-1]} \\
    --n_iters {cfg.n_iters} \\
    --dtype float32 \\
    --high_precision float32 \\
    --save_path $EXPDIR"""
        )
        if level_id == 1 and cfg.subset_indices_path is not None:
            f.write(f" \\\n    --subset_indices_path {cfg.subset_indices_path}\n")
        else:
            f.write("\n")

    with open(Path(level_dir, "local_split_clusters_script.sh"), "w") as f:
        f.write(
f"""#!/usr/bin/env bash

EXPDIR={level_dir}
cd {scripts_path}

PYTHONPATH=.. \\
torchrun \\
--nnodes={cfg.nnodes[level_id-1]} \\
--nproc_per_node={cfg.ngpus_per_node[level_id-1]} \\
    split_clusters.py \\
    --data_path {data_path} \\
    --clusters_path "$EXPDIR"/pre_clusters/sorted_clusters.npy \\
    --n_splits {cfg.n_splits[level_id-1]} \\
    --n_iters {cfg.n_iters} \\
    --dtype float32 \\
    --high_precision float32 \\
    --save_path $EXPDIR"""
        )
        if level_id == 1 and cfg.subset_indices_path is not None:
            f.write(f" \\\n    --subset_indices_path {cfg.subset_indices_path}\n")
        else:
            f.write("\n")


def write_slurm_scripts(cfg):
    """
    Write slurm scripts for all levels.
    """
    for level_id in range(1, cfg.n_levels + 1):
        if cfg.n_splits[level_id - 1] > 1 and cfg.n_resampling_steps[level_id - 1] > 1:
            raise ValueError("Cannot use cluster_split and resampling simultaneously")
        level_dir = Path(cfg.exp_dir, f"level{level_id}").resolve()
        level_dir.mkdir(exist_ok=True)
        Path(level_dir, "logs").mkdir(exist_ok=True)

        write_main_script(cfg, level_dir, level_id, cfg.scripts_path)
        if cfg.n_splits[level_id - 1] > 1:
            write_split_clusters_script(cfg, level_dir, level_id)


def write_launcher(exp_dir, n_levels, n_splits, scripts_path):
    """
    Write bash script to launch slurm scripts in all levels.
    """
    exp_dir = Path(exp_dir).resolve()
    with open(Path(exp_dir, "launcher.sh"), "w") as f:
        f.write(
            f"ID=$(sbatch --parsable {str(exp_dir)}/level1/slurm_script.s | tail -1)\n"
        )
        f.write('echo "Level 1: job $ID"\n')
        if n_splits[0] > 1:
            f.write(
                f'ID=$(sbatch --parsable --dependency=afterok:"$ID" {str(exp_dir)}/level1/slurm_split_clusters_script.s | tail -1)\n'
            )
            f.write('echo "Level 1, split clusters: job $ID"\n')

        for level_id in range(2, n_levels + 1):
            f.write(
                f'ID=$(sbatch --parsable --dependency=afterok:"$ID" {str(exp_dir)}/level{level_id}/slurm_script.s | tail -1)\n'
            )
            f.write(f'echo "Level {level_id}: job $ID"\n')
            if n_splits[level_id - 1] > 1:
                f.write(
                    f'ID=$(sbatch --parsable --dependency=afterok:"$ID" {str(exp_dir)}/level{level_id}/slurm_split_clusters_script.s | tail -1)\n'
                )
                f.write('echo "Level {level_id}, split clusters: job $ID"\n')

def write_local_launcher(exp_dir, n_levels, n_splits, scripts_path):
    """
    Write bash script to launch slurm scripts in all levels.
    """
    exp_dir = Path(exp_dir).resolve()
    with open(Path(exp_dir, "local_launcher.sh"), "w") as f:
        f.write("set -e\n")
        for level_id in range(1, n_levels + 1):
            f.write(f"bash {str(exp_dir)}/level{level_id}/local_script.sh\n")
            if n_splits[level_id - 1] > 1:
                f.write(f"bash {str(exp_dir)}/level{level_id}/local_split_clusters_script.sh\n")


if __name__ == "__main__":
    MLFLOW_EXPERIMENT_NAME = "clustering"
    MLFLOW_RUN_NAME = "clustering"
    ROOT_FOLDER = "outputs/clusters"
    OUTPUT_FOLDER_BASENAME = "clusters"
    METADATA_FILE_BASENAME = "metadata.json"

    parser = ArgumentParser()
    parser.add_argument("--features_id", type=str, required=True)
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--scripts_path", type=str, help="Path to scripts file")
    parser.add_argument("--slurm_account", type=str, help="Account name")
    parser.add_argument("--slurm_partition", type=str, help="Nodes partition")
    parser.add_argument("--slurm_qos", type=str, help="Queue of service")
    parser.add_argument("--slurm_time", type=str, help="Time limit")

    args, opts = parser.parse_known_args()
    print(f"args: {args}")
    print(f"opts: {opts}")
    config_file = args.config_file
    del args.config_file
    if config_file:
        cfg = OmegaConf.load(config_file)
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create(vars(args)),
            OmegaConf.from_cli(opts),
        )
    else:
        cfg = OmegaConf.create(vars(args))
    print(OmegaConf.to_yaml(cfg))
    readable_name_params = {}
    readable_name_params["n_levels"] = str(cfg.n_levels) # abbreviate
    for level in range(cfg.n_levels):
        readable_name_params[f"level{level}"] = str(cfg.n_clusters[level]) # sanitize    

    print("Hierarchical k-means config:")
    print(OmegaConf.to_yaml(cfg))

    config_hash = compute_config_hash(OmegaConf.to_container(cfg, resolve=True))
    cfg.timestamp = datetime.now().isoformat()
    cfg.config_hash = config_hash
    cfg.root_folder = ROOT_FOLDER
    cfg.exp_dir = os.path.join(ROOT_FOLDER, format_readable_run_name(OUTPUT_FOLDER_BASENAME, readable_name_params, config_hash))
    print(cfg.exp_dir)

    """
    
    if os.path.exists(cfg.exp_dir):
        print(f"Clustering già presente {cfg.exp_dir}, cancellare la cartella per ricomputare")
        exit()
    else:
        os.makedirs(cfg.exp_dir)

    try:
        run = mlflow.get_run(cfg.features_id) # a2699b08925941bfb84c8aafd0ab3098
        features_root_folder = run.data.params.get("root_folder")
        features_run_folder = run.data.params.get("run_folder")
        features_info_filename = run.data.params.get("features_filename")
        print(features_root_folder)
        print(features_run_folder)
        print(features_info_filename)
        cfg.embeddings_path = os.path.join(features_root_folder, features_run_folder, features_info_filename)

    except mlflow.exceptions.MlflowException as e:
        print(f"Errore nel recuperare il cutout_id {cfg.features_id}: {e}")

    write_slurm_scripts(cfg)
    write_local_launcher(cfg.exp_dir, cfg.n_levels, cfg.n_splits, cfg.scripts_path)
    write_launcher(cfg.exp_dir, cfg.n_levels, cfg.n_splits, cfg.scripts_path)

    Path(cfg["exp_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["exp_dir"], "config.yaml"), "w") as fid:
        OmegaConf.save(config=cfg, f=fid)
    """
    run_path = cfg.exp_dir
    
    metadata_file_path = os.path.join(run_path, METADATA_FILE_BASENAME)
    
    log_on_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, OmegaConf.to_container(cfg, resolve=True), metadata_file_path)