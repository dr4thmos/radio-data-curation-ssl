# -*- coding: utf-8 -*-
"""
VISUALIZZAZIONE UMAP (UNIFICATO)

Scopo:
Questo script esegue una riduzione dimensionale tramite UMAP e ne visualizza i risultati.
È progettato per funzionare in due modalità, selezionate tramite argomenti:
1.  Modalità "Subsample": Lavora su un sottoinsieme di embedding, definito da un file di indici.
2.  Modalità "Full": Lavora sull'intero set di features letto da un file HDF5.

Entrambe le modalità sono integrate con MLflow per recuperare i percorsi ai dati
e per loggare i risultati generati.

Funzionamento:
1.  Riceve come input un ID di run MLflow (`--subsample_id` o `--features_id`).
2.  In base all'argomento fornito, determina la modalità operativa.
3.  Interroga MLflow per recuperare i parametri e i percorsi ai file necessari.
4.  Carica i dati (l'intero set o un sottoinsieme) e li sposta su GPU.
5.  Applica UMAP (accelerato su GPU tramite cuML).
6.  Genera, salva e opzionalmente logga su MLflow diversi artefatti.

Output Generati:
- Scatter plot della proiezione UMAP (.png).
- Mappa di calore della densità della proiezione UMAP (.png).
- L'embedding 2D calcolato da UMAP (.npy).
- Un file con i parametri usati per l'esecuzione (.json).

Esempio di utilizzo:
# Modalità Subsample
python this_script.py --subsample_id <ID_RUN_SUBSAMPLE> --log_to_mlflow

# Modalità Full
python this_script.py --features_id <ID_RUN_FEATURES> --log_to_mlflow
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import mlflow
import json
import tempfile
import h5py
import cupy as cp
from cuml.manifold.umap import UMAP

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UMAP visualization script (handles full or subsampled data)')
    
    ## UNIFICAZIONE: Gruppo di argomenti che si escludono a vicenda.
    # L'utente DEVE fornire o --features_id o --subsample_id, ma non entrambi.
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--features_id', type=str,
                            help='MLflow run ID for the run containing the FULL HDF5 features file path.')
    mode_group.add_argument('--subsample_id', type=str,
                            help='MLflow run ID for the SUBSAMPLE run containing parameters and indices filename.')

    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved images')
    parser.add_argument('--log_to_mlflow', action='store_true', help='Log results back to the source MLflow run')
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow tracking URI')
    parser.add_argument('--n_neighbors', type=int, default=50, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='euclidean', help='UMAP distance metric')
    return parser.parse_args()

def get_mlflow_data_config(args, client):
    """
    Retrieve data paths and configuration from MLflow based on the provided run ID.
    This function acts as a dispatcher for the different modes.
    """
    ## UNIFICAZIONE: Dispatcher per la configurazione dei dati.
    if args.features_id:
        print(f"--- Mode: FULL --- (Run ID: {args.features_id})")
        return get_config_for_full_features(args.features_id, client)
    elif args.subsample_id:
        print(f"--- Mode: SUBSAMPLE --- (Run ID: {args.subsample_id})")
        return get_config_for_subsample(args.subsample_id, client)

def get_config_for_full_features(features_id, client):
    """Retrieve HDF5 file path for the 'full' mode."""
    print(f"Retrieving parameters for features run ID: {features_id}")
    try:
        run = client.get_run(features_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve run {features_id}. Error: {e}")

    root_folder = run.data.params.get("root_folder")
    run_folder = run.data.params.get("run_folder")
    features_filename = run.data.params.get("features_filename")

    if not all([root_folder, run_folder, features_filename]):
        raise ValueError(f"Missing one or more required params (root_folder, run_folder, features_filename) in run {features_id}")

    hdf5_path = os.path.join(root_folder, run_folder, features_filename)
    return {
        "mode": "full",
        "hdf5_path": hdf5_path,
        "run_id_to_log": features_id,
        "exp_dir": os.path.join(root_folder, run_folder)
    }

def get_config_for_subsample(subsample_id, client):
    """Retrieve embedding and indices paths for the 'subsample' mode."""
    print(f"Retrieving parameters for subsample run ID: {subsample_id}")
    try:
        subsample_run = client.get_run(subsample_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve run {subsample_id}. Error: {e}")

    # Assuming the subsample run has a tag or param pointing to the 'features' run.
    # Let's check for a tag first, then a param.
    features_id = subsample_run.data.tags.get("mlflow.parentRunId") or subsample_run.data.params.get("features_run_id")
    if not features_id:
        raise ValueError(f"Subsample run {subsample_id} must have a 'mlflow.parentRunId' tag or 'features_run_id' param linking to the original features run.")

    indices_filename = subsample_run.data.params.get("subsampled_indices_filename")
    if not indices_filename:
        raise ValueError(f"Missing 'subsampled_indices_filename' in run {subsample_id}")

    # Get info from the parent features run
    try:
        features_run = client.get_run(features_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve parent features run {features_id}. Error: {e}")

    root_folder = features_run.data.params.get("root_folder")
    run_folder = features_run.data.params.get("run_folder")
    features_filename = features_run.data.params.get("features_filename")
    if not all([root_folder, run_folder, features_filename]):
        raise ValueError(f"Missing required params in parent features run {features_id}")

    # The subsampled indices are artifacts of the subsample run itself
    subsample_exp_dir = subsample_run.data.params.get("exp_dir")
    if not subsample_exp_dir:
        raise ValueError(f"Missing 'exp_dir' in subsample run {subsample_id}")

    hdf5_path = os.path.join(root_folder, run_folder, features_filename)
    indices_path = os.path.join(subsample_exp_dir, indices_filename)

    return {
        "mode": "subsample",
        "hdf5_path": hdf5_path,
        "indices_path": indices_path,
        "run_id_to_log": subsample_id,
        "exp_dir": subsample_exp_dir # Log artifacts relative to the subsample run's dir
    }

def load_data_from_config(config):
    """Load data based on the configuration dictionary."""
    mode = config["mode"]
    print(f"\n--- Loading Data ({mode} mode) ---")
    
    hdf5_path = config["hdf5_path"]
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 features file not found at: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as h5f:
        if 'features' not in h5f:
            raise KeyError("Dataset 'features' not found in HDF5 file.")
        features_dset = h5f['features']

        if mode == "full":
            print(f"Loading full features from HDF5. Shape: {features_dset.shape}")
            features_np = features_dset[:]
            return features_np

        elif mode == "subsample":
            indices_path = config["indices_path"]
            if not os.path.exists(indices_path):
                raise FileNotFoundError(f"Indices file not found: {indices_path}")

            print(f"Loading subsampled indices from: {indices_path}")
            sampled_indices = np.load(indices_path)

            if sampled_indices.max() >= features_dset.shape[0] or sampled_indices.min() < 0:
                raise ValueError("Indices are out of bounds for the features array.")
            
            print(f"Applying subsampling. Number of points: {len(sampled_indices)}")
            # Load only the required indices from HDF5 for memory efficiency
            # Sorting indices is required for h5py advanced indexing
            sorted_indices = np.sort(sampled_indices)
            return features_dset[sorted_indices]
        else:
            raise ValueError(f"Unknown data loading mode: {mode}")

def setup_output_directory(exp_dir, run_id, mode):
    """Create a unique output directory."""
    # Using a hash of the run_id to keep the folder name shorter and cleaner
    run_id_hash = hash(run_id) & 0xffffffff 
    output_dir = os.path.join(exp_dir, f"umap_visualization_{mode}_{run_id_hash:x}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

def run_umap_visualization(features_gpu, output_dir, n_neighbors, min_dist, metric, log_to_mlflow, client, run_id, dpi):
    """
    Core UMAP processing and visualization function.
    This function is generic and does not depend on the loading mode.
    """
    try:
        print(f"\n--- Starting UMAP ---")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        print(f"Data shape for UMAP: {features_gpu.shape}")

        umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                            n_components=2, random_state=42, verbose=True)
        
        umap_result_gpu = umap_reducer.fit_transform(features_gpu)
        print(f"UMAP transformation complete. Result shape: {umap_result_gpu.shape}")
        
        # Convert to CPU for plotting and saving with numpy
        umap_result_cpu = cp.asnumpy(umap_result_gpu)
        
        # Define number of points once for reuse
        num_points = umap_result_cpu.shape[0]

        # --- Save UMAP Embedding ---
        embedding_filename = "umap_embedding.npy"
        embedding_path = os.path.join(output_dir, embedding_filename)
        np.save(embedding_path, umap_result_cpu)
        print(f"Saved UMAP embedding to: {embedding_path}")
        if log_to_mlflow:
             client.log_artifact(run_id, embedding_path, artifact_path="umap_visualizations")
        
        # --- Create UMAP Scatter Plot ---
        print("Creating UMAP scatter plot...")
        fig_scatter = plt.figure(figsize=(12, 10))
        plt.scatter(
            umap_result_cpu[:, 0], umap_result_cpu[:, 1],
            s=max(1.0, 200 / np.sqrt(num_points)), # Use num_points
            alpha=0.7, c='steelblue', edgecolors='none'
        )
        plt.title(f'UMAP Projection (n={num_points}, n_neighbors={n_neighbors}, min_dist={min_dist})') # Use num_points
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        scatter_plot_filename = "umap_projection_scatter.png"
        scatter_plot_path = os.path.join(output_dir, scatter_plot_filename)
        fig_scatter.savefig(scatter_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved UMAP scatter plot to: {scatter_plot_path}")
        if log_to_mlflow:
            client.log_artifact(run_id, scatter_plot_path, artifact_path="umap_visualizations")
        plt.close(fig_scatter)
        
        # --- Create UMAP Density Heatmap ---
        try:
            print("Creating UMAP density heatmap...")
            from scipy.stats import gaussian_kde

            if num_points > 100000: # Limita per evitare problemi di memoria/tempo
                 print(f"Dataset > 100k points ({num_points}), skipping density heatmap generation for performance.")
            else:
                # Use umap_result_cpu, which is a NumPy array
                kde = gaussian_kde(umap_result_cpu.T)

                # Use umap_result_cpu for calculating bounds
                x_min, x_max = np.percentile(umap_result_cpu[:, 0], [1, 99])
                y_min, y_max = np.percentile(umap_result_cpu[:, 1], [1, 99])
                padding_x = (x_max - x_min) * 0.1
                padding_y = (y_max - y_min) * 0.1

                x_grid, y_grid = np.meshgrid(
                    np.linspace(x_min - padding_x, x_max + padding_x, 150),
                    np.linspace(y_min - padding_y, y_max + padding_y, 150)
                )
                positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                density = kde(positions).reshape(x_grid.shape)

                fig_density = plt.figure(figsize=(13, 10))
                pcm = plt.pcolormesh(x_grid, y_grid, density, shading='auto', cmap='viridis')
                plt.colorbar(pcm, label='Estimated Density')
                # Use num_points in the title
                plt.title(f'UMAP Density (n={num_points}, n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16)
                plt.xlabel('UMAP Dimension 1', fontsize=12)
                plt.ylabel('UMAP Dimension 2', fontsize=12)
                plt.xlim(x_grid.min(), x_grid.max())
                plt.ylim(y_grid.min(), y_grid.max())
                plt.tight_layout()

                density_plot_filename = "umap_projection_density.png"
                density_plot_path = os.path.join(output_dir, density_plot_filename)
                fig_density.savefig(density_plot_path, dpi=dpi, bbox_inches='tight')
                print(f"Saved UMAP density plot to: {density_plot_path}")

                if log_to_mlflow and client and run_id:
                    client.log_artifact(run_id, density_plot_path, artifact_path="umap_visualizations")
                plt.close(fig_density)

        except ImportError:
             print("Scipy not installed, skipping density heatmap generation.")
        except Exception as e:
            print(f"Warning: Could not create density plot. Error: {e}")


        # --- Save UMAP Parameters ---
        # All variables are now in the correct scope
        umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "input_shape": list(features_gpu.shape),
            "embedding_shape": list(umap_result_cpu.shape),
            "num_points_processed": num_points
        }
        params_filename = "umap_run_parameters.json"
        params_path = os.path.join(output_dir, params_filename)
        with open(params_path, 'w') as f:
            json.dump(umap_params, f, indent=4)
        print(f"Saved UMAP parameters to: {params_path}")

        if log_to_mlflow and client and run_id:
            try:
                client.log_param(run_id, "umap_n_neighbors", n_neighbors)
                client.log_param(run_id, "umap_min_dist", min_dist)
                client.log_param(run_id, "umap_metric", metric)
                client.log_artifact(run_id, params_path, artifact_path="umap_visualizations")
                print(f"Logged UMAP parameters and {params_filename} to MLflow run {run_id}")
            except Exception as e:
                print(f"Error logging UMAP parameters/file to MLflow: {e}")

        print("--- UMAP Finished Successfully ---")
        return umap_result_cpu

    except Exception as e:
        print(f"Error during UMAP processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_arguments()

    try:
        # Set up MLflow
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # 1. Get data configuration from MLflow
        data_config = get_mlflow_data_config(args, client)
        
        # 2. Setup output directory
        output_dir = setup_output_directory(data_config["exp_dir"], data_config["run_id_to_log"], data_config["mode"])

        # 3. Load data based on config
        features_np = load_data_from_config(data_config)
        
        # 4. Move data to GPU
        print("\n--- Preparing Data for GPU ---")
        mem_gb = features_np.nbytes / (1024**3)
        print(f"Data size: {mem_gb:.2f} GB. Moving to GPU...")
        features_gpu = cp.array(features_np)
        del features_np # Free up CPU memory
        print("Data successfully moved to GPU.")

        # 5. Run UMAP
        run_umap_visualization(
            features_gpu,
            output_dir,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            log_to_mlflow=args.log_to_mlflow,
            client=client if args.log_to_mlflow else None,
            run_id=data_config["run_id_to_log"] if args.log_to_mlflow else None,
            dpi=args.dpi
        )
        
        print("\n--- UMAP Visualization Process Completed Successfully ---")

    except (FileNotFoundError, ValueError, KeyError) as e:
         print(f"\nERROR: A configuration or file error occurred: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()