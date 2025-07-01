# -*- coding: utf-8 -*-
"""
Script per visualizzazione UMAP (GPU) su dati sottocampionati via MLflow.

- Input: `subsample_id` di un run MLflow.
- Logica: Trova i path degli embedding e degli indici tramite MLflow,
  carica i dati e applica UMAP (cuML) sul sottoinsieme.
- Output: Salva e/o logga su MLflow:
    1. Scatter plot (.png)
    2. Density map (.png)
    3. Embedding UMAP (.npy)
    4. Parametri usati (.json)
"""



import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd # Non sembra più utilizzato, possiamo rimuoverlo
import os
import argparse
import mlflow
import json
import tempfile
from cuml.manifold.umap import UMAP
# from sklearn.decomposition import PCA # Non più necessario senza la parte di test

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UMAP visualization of subsampled data using MLflow parameters')
    parser.add_argument('--subsample_id', type=str, required=True,
                        help='MLflow run ID for the subsample run containing parameters and indices filename.')
    # Rimosso --test_mode
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved images (default: 300)')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Log results back to the subsample MLflow run')
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLflow tracking URI (optional)')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='UMAP n_neighbors parameter (default: 50)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='UMAP distance metric (default: euclidean)')
    return parser.parse_args()

def get_mlflow_parameters(subsample_id, client):
    """Retrieve parameters from MLflow based on subsample_id."""
    print(f"Retrieving parameters for subsample run ID: {subsample_id}")

    # Get subsample run
    try:
        subsample_run = client.get_run(subsample_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve run {subsample_id} from MLflow. Error: {e}")

    # Extract parameters from subsample run
    clusters_id = subsample_run.data.params.get("clusters_id")
    subsampled_indices_filename = subsample_run.data.params.get("subsampled_indices_filename")

    if not clusters_id:
        raise ValueError(f"Could not find 'clusters_id' parameter in run {subsample_id}")
    if not subsampled_indices_filename:
        raise ValueError(f"Could not find 'subsampled_indices_filename' parameter in run {subsample_id}")

    print(f"Found clusters_id: {clusters_id}")
    print(f"Found subsampled_indices_filename: {subsampled_indices_filename}")

    # Get clusters run (parent run containing paths)
    try:
        clusters_run = client.get_run(clusters_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve clusters run {clusters_id} from MLflow. Error: {e}")

    # Extract parameters from clusters run
    embeddings_path = clusters_run.data.params.get("embeddings_path")
    exp_dir = clusters_run.data.params.get("exp_dir") # Directory base esperimento

    if not embeddings_path:
        raise ValueError(f"Could not find 'embeddings_path' parameter in clusters run {clusters_id}")
    if not exp_dir:
         raise ValueError(f"Could not find 'exp_dir' parameter in clusters run {clusters_id}")

    print(f"Found embeddings_path: {embeddings_path}")
    print(f"Found exp_dir: {exp_dir}")

    # Construct full path to indices file relative to exp_dir
    # Assumiamo che subsampled_indices_filename sia solo il nome del file
    # e che si trovi dentro la directory specificata da exp_dir
    # Se il filename contenesse già un path relativo/assoluto, adattare la logica
    indices_path = os.path.join(exp_dir, subsampled_indices_filename)
    print(f"Constructed indices_path: {indices_path}")


    return {
        "clusters_id": clusters_id,
        "subsampled_indices_filename": subsampled_indices_filename, # Nome file indici
        "indices_path": indices_path, # Path completo indici
        "embeddings_path": embeddings_path, # Path completo embeddings originali
        "exp_dir": exp_dir # Directory base esperimento
    }

def setup_output_directory(exp_dir, subsample_id):
    """Create output directory based on exp_dir and subsample_id."""
    # Crea una sottodirectory specifica per i risultati UMAP di questo subsample_id
    output_dir = os.path.join(exp_dir, f"umap_visualization_{subsample_id}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

def log_figure_to_mlflow(fig, artifact_filename, client, run_id, artifact_path="umap_visualizations"):
    """Save figure temporarily and log to MLflow."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
        fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
        try:
            client.log_artifact(run_id, tmp.name, artifact_path=artifact_path)
            print(f"Logged {artifact_filename} to MLflow run {run_id} in path {artifact_path}")
        except Exception as e:
            print(f"Error logging artifact {artifact_filename} to MLflow: {e}")

def run_umap_visualization(features, sampled_indices, output_dir, n_neighbors=50, min_dist=0.1, metric='euclidean',
             log_to_mlflow=False, client=None, run_id=None, dpi=300):
    """Run UMAP on the data and create visualizations."""
    try:
        print(f"\n--- Starting UMAP ---")
        print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        print(f"Data shape for UMAP: {features[sampled_indices].shape}")

        # Initialize UMAP using cuML
        umap_reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2, # Standard 2D visualization
            random_state=42,
            verbose=True # Aggiunge un po' di output durante l'esecuzione
        )

        # Fit and transform on subsampled data
        print(f"Running UMAP reduction on {len(sampled_indices)} points...")
        umap_result = umap_reducer.fit_transform(features[sampled_indices])
        print(f"UMAP transformation complete. Result shape: {umap_result.shape}")

        # --- Save UMAP Embedding ---
        embedding_filename = "umap_embedding.npy"
        embedding_path = os.path.join(output_dir, embedding_filename)
        np.save(embedding_path, umap_result)
        print(f"Saved UMAP embedding to: {embedding_path}")

        if log_to_mlflow and client and run_id:
             try:
                 client.log_artifact(run_id, embedding_path, artifact_path="umap_visualizations")
                 print(f"Logged {embedding_filename} to MLflow run {run_id}")
             except Exception as e:
                 print(f"Error logging embedding artifact to MLflow: {e}")


        # --- Create UMAP Scatter Plot ---
        print("Creating UMAP scatter plot...")
        fig_scatter = plt.figure(figsize=(12, 10))
        plt.scatter(
            umap_result[:, 0],
            umap_result[:, 1],
            s=max(1.0, 200 / np.sqrt(len(sampled_indices))), # Dimensione punto adattiva
            alpha=0.7,
            c='steelblue', # Colore base, si potrebbe mappare a qualche feature se disponibile
            edgecolors='none' # Rimuove i bordi per plot più puliti con molti punti
        )
        plt.title(f'UMAP Projection (n={len(sampled_indices)}, n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save scatter plot locally
        scatter_plot_filename = "umap_projection_scatter.png"
        scatter_plot_path = os.path.join(output_dir, scatter_plot_filename)
        fig_scatter.savefig(scatter_plot_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved UMAP scatter plot to: {scatter_plot_path}")

        # Log scatter plot to MLflow
        if log_to_mlflow and client and run_id:
            log_figure_to_mlflow(fig_scatter, scatter_plot_filename, client, run_id, artifact_path="umap_visualizations")
        plt.close(fig_scatter) # Chiudi la figura per liberare memoria


        # --- Create UMAP Density Heatmap (Optional but useful) ---
        try:
            print("Creating UMAP density heatmap...")
            from scipy.stats import gaussian_kde

            # Calcola la stima della densità kernel (Kernel Density Estimate)
            # Nota: può essere lento/memory intensive per N molto grandi
            if umap_result.shape[0] > 100000: # Limita per evitare problemi di memoria/tempo
                 print("Dataset > 100k points, skipping density heatmap generation for performance.")
            else:
                kde = gaussian_kde(umap_result.T)

                # Crea una griglia per la valutazione della densità
                x_min, x_max = np.percentile(umap_result[:, 0], [1, 99]) # Usa percentili per robustezza agli outlier
                y_min, y_max = np.percentile(umap_result[:, 1], [1, 99])
                padding_x = (x_max - x_min) * 0.1
                padding_y = (y_max - y_min) * 0.1

                x_grid, y_grid = np.meshgrid(
                    np.linspace(x_min - padding_x, x_max + padding_x, 150), # Risoluzione griglia
                    np.linspace(y_min - padding_y, y_max + padding_y, 150)
                )
                positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

                # Valuta KDE sulla griglia
                density = kde(positions).reshape(x_grid.shape)

                # Crea il plot della densità
                fig_density = plt.figure(figsize=(13, 10)) # Leggermente più piccola della scatter
                pcm = plt.pcolormesh(x_grid, y_grid, density, shading='auto', cmap='viridis') # 'viridis' è un buon cmap
                plt.colorbar(pcm, label='Estimated Density')
                plt.title(f'UMAP Density (n={len(sampled_indices)}, n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16)
                plt.xlabel('UMAP Dimension 1', fontsize=12)
                plt.ylabel('UMAP Dimension 2', fontsize=12)
                plt.xlim(x_grid.min(), x_grid.max())
                plt.ylim(y_grid.min(), y_grid.max())
                plt.tight_layout()

                # Save density plot locally
                density_plot_filename = "umap_projection_density.png"
                density_plot_path = os.path.join(output_dir, density_plot_filename)
                fig_density.savefig(density_plot_path, dpi=dpi, bbox_inches='tight')
                print(f"Saved UMAP density plot to: {density_plot_path}")

                # Log density plot to MLflow
                if log_to_mlflow and client and run_id:
                    log_figure_to_mlflow(fig_density, density_plot_filename, client, run_id, artifact_path="umap_visualizations")
                plt.close(fig_density) # Chiudi figura

        except ImportError:
             print("Scipy not installed, skipping density heatmap generation.")
        except Exception as e:
            print(f"Warning: Could not create density plot. Error: {e}")


        # --- Save UMAP Parameters ---
        umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "input_shape": list(features[sampled_indices].shape),
            "embedding_shape": list(umap_result.shape),
            "subsample_size": len(sampled_indices)
        }
        params_filename = "umap_run_parameters.json"
        params_path = os.path.join(output_dir, params_filename)
        with open(params_path, 'w') as f:
            json.dump(umap_params, f, indent=4)
        print(f"Saved UMAP parameters to: {params_path}")

        # Log parameters to MLflow
        if log_to_mlflow and client and run_id:
            try:
                # Logga i parametri UMAP specifici usati in questo run
                client.log_param(run_id, "umap_n_neighbors", n_neighbors)
                client.log_param(run_id, "umap_min_dist", min_dist)
                client.log_param(run_id, "umap_metric", metric)
                # Logga anche il file JSON come artifact per completezza
                client.log_artifact(run_id, params_path, artifact_path="umap_visualizations")
                print(f"Logged UMAP parameters and {params_filename} to MLflow run {run_id}")
            except Exception as e:
                print(f"Error logging UMAP parameters/file to MLflow: {e}")

        print("--- UMAP Finished Successfully ---")
        return umap_result

    except Exception as e:
        print(f"Error during UMAP processing: {e}")
        import traceback
        traceback.print_exc()
        print("--- UMAP Failed ---")
        return None

def main():
    # Parse arguments
    args = parse_arguments()

    try:
        # Set up MLflow client
        if args.tracking_uri:
            print(f"Setting MLflow tracking URI to: {args.tracking_uri}")
            mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Get parameters from MLflow runs
        print("\n--- Retrieving MLflow Parameters ---")
        mlflow_params = get_mlflow_parameters(args.subsample_id, client)

        # Setup local output directory
        print("\n--- Setting up Output Directory ---")
        output_dir = setup_output_directory(mlflow_params["exp_dir"], args.subsample_id)

        # Save retrieved MLflow parameters locally for reference
        mlflow_params_path = os.path.join(output_dir, "retrieved_mlflow_parameters.json")
        with open(mlflow_params_path, 'w') as f:
            json.dump(mlflow_params, f, indent=4)
        print(f"Saved retrieved MLflow parameters to: {mlflow_params_path}")

        # Log retrieved parameters back to the subsample run if requested
        if args.log_to_mlflow:
            try:
                client.log_dict(args.subsample_id, mlflow_params, "umap_visualizations/retrieved_mlflow_parameters.json")
                print(f"Logged retrieved parameters to MLflow run {args.subsample_id}")
            except Exception as e:
                print(f"Warning: Could not log retrieved parameters to MLflow. Error: {e}")

        # --- Load Data ---
        print("\n--- Loading Data ---")
        embeddings_path = mlflow_params["embeddings_path"]
        indices_path = mlflow_params["indices_path"]

        if not os.path.exists(embeddings_path):
             raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
        if not os.path.exists(indices_path):
             raise FileNotFoundError(f"Indices file not found at: {indices_path}")

        print(f"Loading embeddings from: {embeddings_path}")
        features = np.load(embeddings_path)
        print(f"Full features shape: {features.shape}")

        print(f"Loading subsampled indices from: {indices_path}")
        sampled_indices = np.load(indices_path)
        print(f"Sampled indices shape: {sampled_indices.shape}")
        print(f"Number of points to process: {len(sampled_indices)}")

        # Sanity check indices
        if sampled_indices.max() >= features.shape[0] or sampled_indices.min() < 0:
            raise ValueError("Indices contain values out of bounds for the features array.")
        if len(sampled_indices) == 0:
             raise ValueError("Indices file is empty.")

        # --- Run UMAP Visualization ---
        umap_result = run_umap_visualization(
            features,
            sampled_indices,
            output_dir,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            log_to_mlflow=args.log_to_mlflow,
            client=client if args.log_to_mlflow else None,
            run_id=args.subsample_id if args.log_to_mlflow else None,
            dpi=args.dpi
        )

        if umap_result is not None:
            print("\n--- UMAP Visualization Process Completed Successfully ---")
            print(f"Results saved locally in: {output_dir}")
            if args.log_to_mlflow:
                print(f"Results logged to MLflow run: {args.subsample_id}")
        else:
            print("\n--- UMAP Visualization Process Failed ---")

    except FileNotFoundError as e:
         print(f"Error: Input file not found. {e}")
         import traceback
         traceback.print_exc()
    except ValueError as e:
         print(f"Error: Invalid value or configuration. {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()