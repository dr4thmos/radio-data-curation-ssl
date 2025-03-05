import itertools
import numpy as np
import yaml
import random

def compute_total_resampled_images(n_levels, n_clusters, sample_size, n_resampling_steps):
    total_images = 0
    for level in range(n_levels):
        k = n_clusters[level]
        r = sample_size[level]
        m = n_resampling_steps[level]
        total_images += k * r * m
    return total_images

def generate_diverse_configurations(target_images, level_range=(1, 5), cluster_range=(5, 100), sample_range=(10, 2000), steps_range=(1, 20), num_configs=10):
    configurations = []

    for _ in range(num_configs):
        n_levels = random.randint(level_range[0], level_range[1])

        n_clusters = [random.randint(cluster_range[0], cluster_range[1]) for _ in range(n_levels)]
        sample_size = [random.randint(sample_range[0], sample_range[1]) for _ in range(n_levels)]
        n_resampling_steps = [random.randint(steps_range[0], steps_range[1]) for _ in range(n_levels)]

        total_images = compute_total_resampled_images(n_levels, n_clusters, sample_size, n_resampling_steps)

        configurations.append({
            "n_levels": n_levels,
            "n_clusters": n_clusters,
            "sample_size": sample_size,
            "n_resampling_steps": n_resampling_steps,
            "total_images": total_images
        })

    # Sort configurations by how close they are to the target
    configurations = sorted(configurations, key=lambda x: abs(x["total_images"] - target_images))
    return configurations[:num_configs]

def save_configurations_to_yaml(configurations, output_dir="configs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, config in enumerate(configurations):
        config_data = {
            "n_levels": config["n_levels"],
            "n_clusters": config["n_clusters"],
            "n_resampling_steps": config["n_resampling_steps"],
            "sample_size": config["sample_size"],
            "dtype": "float64",
            "high_precision": "float64",
            "checkpoint_period": 1000,
            "subset_indices_path": None,
            "ngpus_per_node": [1] * config["n_levels"],
            "nnodes": [1] * config["n_levels"],
            "ncpus_per_gpu": 8,
            "sampling_strategy": "c",
            "slurm_partition": None
        }

        file_path = os.path.join(output_dir, f"config_{i + 1}.yaml")
        with open(file_path, "w") as file:
            yaml.dump(config_data, file)

    print(f"Saved {len(configurations)} configurations to {output_dir}")

# Set the target number of images
TARGET_IMAGES = 1_000_000

# Generate diverse configurations
configs = generate_diverse_configurations(
    target_images=TARGET_IMAGES,
    level_range=(2, 4),  # Range of levels (e.g., 2 to 5)
    cluster_range=(5, 200000),
    sample_range=(10, 2000),
    steps_range=(5, 10),
    num_configs=10  # Generate 10 diverse configurations
)

# Save configurations to YAML files
save_configurations_to_yaml(configs, output_dir="configs")
