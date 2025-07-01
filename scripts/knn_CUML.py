# precompute_all_neighbors_cuml.py
import numpy as np
import cuml
from cuml.neighbors import NearestNeighbors
import time
import argparse
import os
import mlflow
import traceback
import math # Per calcolare i batch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Precompute and save all nearest neighbors using cuML and MLflow.')
    # MLflow Args
    parser.add_argument('--feature_run_id', type=str, required=True,
                        help='MLflow run ID for the run containing the full embeddings path.')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Log the neighbor arrays and parameters back to the feature MLflow run.')
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLflow tracking URI (optional).')
    # Computation Args
    parser.add_argument('--k', type=int, default=100, # Salvare più vicini è utile per analisi future
                        help='Number of nearest neighbors to compute and save for each point.')
    parser.add_argument('--metric', type=str, default='cosine', choices=['euclidean', 'cosine'],
                        help='Distance metric to use (default: cosine).')
    parser.add_argument('--batch_size', type=int, default=32768, # Batch per kneighbors
                        help='Batch size for querying neighbors to manage memory (default: 32768). Adjust based on GPU memory.')
    # Output Args
    parser.add_argument('--output_dir_suffix', type=str, default='all_neighbors',
                        help='Suffix for the output directory within exp_dir.')
    parser.add_argument('--indices_filename', type=str, default='neighbor_indices.npy',
                        help='Filename for the saved neighbor indices array.')
    parser.add_argument('--distances_filename', type=str, default='neighbor_distances.npy',
                        help='Filename for the saved neighbor distances array.')
    parser.add_argument('--save_dtype_indices', type=str, default='int32', choices=['int32', 'int64'],
                        help='Data type for saving indices (int32 saves space if N < 2^31).')
    return parser.parse_args()

def get_mlflow_params(feature_run_id, client):
    """Retrieve necessary parameters from MLflow."""
    print(f"Retrieving parameters from feature run ID: {feature_run_id}")
    try:
        feature_run = client.get_run(feature_run_id)
    except Exception as e:
        print(f"ERROR: Could not retrieve run {feature_run_id}. Error: {e}")
        raise ValueError(f"Could not retrieve run {feature_run_id}. Error: {e}")

    embeddings_path = feature_run.data.params.get("embeddings_path")
    exp_dir = feature_run.data.params.get("exp_dir")
    if not embeddings_path: raise ValueError(f"'embeddings_path' not found in run {feature_run_id}")
    if not exp_dir: raise ValueError(f"'exp_dir' not found in run {feature_run_id}")
    print(f"Found embeddings_path: {embeddings_path}")
    print(f"Found exp_dir: {exp_dir}")
    return {"embeddings_path": embeddings_path, "exp_dir": exp_dir}

def log_artifact_to_mlflow(local_path, client, run_id, artifact_subdir="all_neighbors_results"):
    """Log a local file artifact to MLflow."""
    if not client or not run_id:
        print(f"Warning: Cannot log artifact {local_path} - MLflow client or run_id missing.")
        return
    try:
        artifact_name = os.path.basename(local_path)
        print(f"Attempting to log artifact '{artifact_name}' to run_id '{run_id}' in subdir '{artifact_subdir}'...")
        client.log_artifact(run_id, local_path, artifact_path=artifact_subdir)
        print(f"  Successfully logged artifact '{artifact_name}'")
    except Exception as e:
        print(f"  ERROR logging artifact '{artifact_name}' to MLflow run '{run_id}': {e}")

def main():
    args = parse_arguments()
    print("--- Precompute All Nearest Neighbors Script ---")
    print(f"Arguments: {args}")

    # --- Validate K ---
    if args.k <= 0:
        print("Error: k must be positive.")
        exit(1)

    try:
        # --- MLflow Setup ---
        if args.tracking_uri: mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        print(f"MLflow tracking URI active: {mlflow.get_tracking_uri()}")
        print(f"Logging to MLflow enabled: {args.log_to_mlflow}")
        mlflow_client = client if args.log_to_mlflow else None
        target_run_id = args.feature_run_id # Logghiamo nella run delle features

        print("\n--- Retrieving Parameters via MLflow ---")
        mlflow_params = get_mlflow_params(args.feature_run_id, client)
        features_path = mlflow_params['embeddings_path']
        exp_dir = mlflow_params['exp_dir']

        # --- Setup Output Directory ---
        output_dir = os.path.join(exp_dir, args.output_dir_suffix + f"_k{args.k}_{args.metric}")
        print(f"Output directory for neighbor data: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        indices_path = os.path.join(output_dir, args.indices_filename)
        distances_path = os.path.join(output_dir, args.distances_filename)
        params_path = os.path.join(output_dir, "precompute_params.json") # Per salvare i parametri usati

        # --- Load Features ---
        print(f"\n--- Loading Features ---")
        t0 = time.time()
        if not os.path.exists(features_path): raise FileNotFoundError(f"Features file not found: {features_path}")
        features = np.load(features_path)
        # Preprocessing
        if features.dtype != np.float32:
            print("Converting features to float32.")
            features = features.astype(np.float32)
        if not features.flags['C_CONTIGUOUS']:
             print("Making features C-contiguous.")
             features = np.ascontiguousarray(features)
        load_time = time.time() - t0
        n_total, d = features.shape
        print(f"Loaded {n_total} features (dim={d}) in {load_time:.2f} seconds.")
        if n_total == 0: raise ValueError("Features file is empty.")

        # --- Normalize (if using cosine metric) ---
        norm_time = 0.0
        if args.metric == 'cosine':
            print("\n--- Normalizing Features for Cosine Metric ---")
            t0 = time.time()
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            features /= norms
            norm_time = time.time() - t0
            print(f"Normalization done in {norm_time:.2f} seconds.")

        # --- Setup cuML Nearest Neighbors ---
        print(f"\n--- Setting up cuML NearestNeighbors (k={args.k}, metric='{args.metric}') ---")
        # Chiediamo k vicini (non k+1) perché qui itereremo su *tutti* i punti come query,
        # quindi il punto stesso sarà uno dei risultati se è il più vicino a se stesso (dist=0).
        # Lo gestiremo dopo, se necessario. Se K è grande, è improbabile che il punto stesso
        # rimanga nei top K se non è isolato. Per sicurezza, potremmo chiedere k+1 e rimuovere dopo.
        # Decidiamo per k+1 per sicurezza e rimuoveremo self-match dopo.
        n_neighbors_to_query = args.k + 1
        t0 = time.time()
        nn_model = NearestNeighbors(n_neighbors=n_neighbors_to_query, metric=args.metric)
        print("Fitting NearestNeighbors model...")
        nn_model.fit(features)
        fit_time = time.time() - t0
        print(f"Model fitting done in {fit_time:.2f} seconds.")

        # --- Precompute Neighbors in Batches ---
        print(f"\n--- Precomputing Neighbors for All {n_total} Points (Batch Size: {args.batch_size}) ---")
        print(f"WARNING: This will take time!")

        # Determine output dtypes
        index_dtype = np.int32 if args.save_dtype_indices == 'int32' else np.int64
        if n_total >= 2**31 and args.save_dtype_indices == 'int32':
             print("Warning: N > 2^31, forcing index dtype to int64.")
             index_dtype = np.int64

        # Pre-allocate output arrays (memoria permettendo)
        # Se N è molto grande, questo potrebbe fallire. Alternativa: scrivere su file per batch.
        # Qui assumiamo che N * K * (4+4) o (8+4) bytes stiano in memoria CPU.
        # Per 3M * 100 * 8 bytes = ~2.4 GB (indici) + 3M * 100 * 4 bytes = ~1.2 GB (distanze) -> ~3.6 GB OK
        print(f"Allocating memory for results ({n_total} x {n_neighbors_to_query}, dtype_idx={index_dtype.__name__})...")
        try:
            all_distances = np.zeros((n_total, n_neighbors_to_query), dtype=np.float32)
            all_indices = np.zeros((n_total, n_neighbors_to_query), dtype=index_dtype)
        except MemoryError as e:
            print(f"FATAL ERROR: Cannot allocate memory for results arrays. N={n_total}, K={n_neighbors_to_query}. Error: {e}")
            print("Consider reducing K or implementing batch writing to disk.")
            exit(1)
        print("Memory allocated.")


        total_query_time = 0
        num_batches = math.ceil(n_total / args.batch_size)
        start_batch_time = time.time()

        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, n_total)
            batch_indices = np.arange(start_idx, end_idx) # Indici dei punti da interrogare in questo batch

            print(f"Processing batch {i+1}/{num_batches} (indices {start_idx} to {end_idx-1})...")
            t_batch0 = time.time()

            # Query per il batch corrente
            # Passiamo le features corrispondenti agli indici del batch
            batch_features = features[start_idx:end_idx, :]
            distances_batch, indices_batch = nn_model.kneighbors(batch_features)

            # Salva i risultati negli array preallocati
            all_distances[start_idx:end_idx, :] = distances_batch
            all_indices[start_idx:end_idx, :] = indices_batch

            t_batch1 = time.time()
            batch_time = t_batch1 - t_batch0
            total_query_time += batch_time
            print(f"  Batch {i+1} finished in {batch_time:.2f} seconds.")

            # Stima tempo rimanente (molto approssimativa)
            if i > 0:
                 avg_batch_time = total_query_time / (i + 1)
                 remaining_batches = num_batches - (i + 1)
                 eta_seconds = remaining_batches * avg_batch_time
                 eta_min = eta_seconds / 60
                 print(f"  Estimated time remaining: {eta_min:.1f} minutes.")

        print(f"\nTotal query time for all batches: {total_query_time:.2f} seconds.")

        # --- Post-process (Opzionale: rimuovere self-match se è il primo risultato) ---
        print("\n--- Post-processing Results ---")
        # Verifica se il primo vicino è sempre se stesso (distanza vicina a zero)
        # Prendiamo un campione per verificarlo, non tutti i 3M
        sample_check_indices = np.random.choice(n_total, min(1000, n_total), replace=False)
        self_is_first = np.all(all_indices[sample_check_indices, 0] == sample_check_indices)
        avg_first_dist = np.mean(all_distances[sample_check_indices, 0])

        if self_is_first and avg_first_dist < 1e-5:
            print("Confirmed: First neighbor is the point itself. Removing first column (self-neighbor).")
            # Rimuovi la prima colonna (indice 0) da entrambi gli array
            final_indices = all_indices[:, 1:]
            final_distances = all_distances[:, 1:]
            final_k = args.k # Abbiamo ottenuto k vicini reali
            print(f"Final array shapes: Indices={final_indices.shape}, Distances={final_distances.shape}")
        else:
            print(f"Warning: First neighbor might not always be self (self_is_first={self_is_first}, avg_first_dist={avg_first_dist:.2e}).")
            print(f"Keeping all {n_neighbors_to_query} columns. The effective K might be {args.k} or {args.k+1}.")
            # In questo caso raro, manteniamo tutti i k+1 risultati
            final_indices = all_indices
            final_distances = all_distances
            final_k = args.k + 1 # Potremmo averne k+1


        # --- Save Results ---
        print("\n--- Saving Neighbor Arrays ---")
        t0 = time.time()
        np.save(indices_path, final_indices)
        print(f"Saved neighbor indices to: {indices_path} (shape: {final_indices.shape}, dtype: {final_indices.dtype})")
        np.save(distances_path, final_distances)
        print(f"Saved neighbor distances to: {distances_path} (shape: {final_distances.shape}, dtype: {final_distances.dtype})")
        save_time = time.time() - t0
        print(f"Arrays saved in {save_time:.2f} seconds.")

        # --- Save Parameters Used ---
        computation_params = {
            "feature_run_id": args.feature_run_id,
            "k_requested": args.k,
            "k_results": final_k, # K effettivo salvato
            "metric": args.metric,
            "normalized_features": args.metric == 'cosine', # Assumendo che normalizziamo solo per coseno
            "batch_size": args.batch_size,
            "save_dtype_indices": args.save_dtype_indices,
            "num_vectors": n_total,
            "vector_dimension": d,
            "indices_shape": list(final_indices.shape),
            "distances_shape": list(final_distances.shape),
            "indices_dtype": str(final_indices.dtype),
            "distances_dtype": str(final_distances.dtype),
            "time_load_s": load_time,
            "time_norm_s": norm_time,
            "time_fit_s": fit_time,
            "time_query_total_s": total_query_time,
            "time_save_s": save_time,
            "indices_filesize_mb": os.path.getsize(indices_path) / (1024*1024) if os.path.exists(indices_path) else -1,
            "distances_filesize_mb": os.path.getsize(distances_path) / (1024*1024) if os.path.exists(distances_path) else -1,
        }
        with open(params_path, 'w') as f:
            json.dump(computation_params, f, indent=4)
        print(f"Saved computation parameters to: {params_path}")

        # --- Log to MLflow (if requested) ---
        if args.log_to_mlflow:
            print("\n--- Logging Results to MLflow ---")
            if mlflow_client and target_run_id:
                log_artifact_to_mlflow(indices_path, mlflow_client, target_run_id)
                log_artifact_to_mlflow(distances_path, mlflow_client, target_run_id)
                log_artifact_to_mlflow(params_path, mlflow_client, target_run_id)
                # Log metriche chiave
                mlflow_client.log_param(target_run_id, f"neighbors_k", final_k)
                mlflow_client.log_param(target_run_id, f"neighbors_metric", args.metric)
                mlflow_client.log_metric(target_run_id, f"neighbors_time_query_total_s", total_query_time)
                mlflow_client.log_metric(target_run_id, f"neighbors_indices_filesize_mb", computation_params["indices_filesize_mb"])
                mlflow_client.log_metric(target_run_id, f"neighbors_distances_filesize_mb", computation_params["distances_filesize_mb"])
                print("Logging to MLflow complete.")
            else:
                print("Warning: log_to_mlflow specified, but client or run_id is invalid.")


        print("\n--- Precomputation Script Finished Successfully ---")

    except FileNotFoundError as e:
         print(f"FATAL ERROR: Input file not found. {e}"); traceback.print_exc(); exit(1)
    except ValueError as e:
         print(f"FATAL ERROR: Invalid value or configuration. {e}"); traceback.print_exc(); exit(1)
    except MemoryError as e:
         print(f"FATAL ERROR: Out of memory. {e}"); traceback.print_exc(); exit(1)
    except Exception as e:
        print(f"\n--- An Unexpected FATAL Error Occurred ---"); print(f"Error: {e}"); traceback.print_exc(); exit(1)
    finally:
        print("\n--- Script End ---")

if __name__ == "__main__":
    main()