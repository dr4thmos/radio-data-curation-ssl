# build_faiss_index.py (versione integrata con MLflow)
import numpy as np
import faiss
import time
import argparse
import os
import mlflow
import traceback
import json # Per salvare i parametri

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build and save a FAISS index using MLflow parameters.')
    # MLflow Args
    parser.add_argument('--feature_run_id', type=str, required=True,
                        help='MLflow run ID for the run containing the full embeddings path.')
    parser.add_argument('--log_to_mlflow', action='store_true',
                        help='Log the FAISS index and parameters back to the feature MLflow run.')
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLflow tracking URI (optional).')
    # FAISS Args
    parser.add_argument('--output_index_filename', type=str, default='faiss_index.index',
                        help='Filename for the saved FAISS index (will be placed in exp_dir).')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for index building.')
    parser.add_argument('--normalize', action='store_true', help='L2-normalize features before indexing (recommended for IndexFlatIP).')
    parser.add_argument('--index_type', type=str, default='IndexFlatL2', choices=['IndexFlatL2', 'IndexFlatIP'],
                        help='Type of FAISS index to build (default: IndexFlatL2).')
    return parser.parse_args()

def get_mlflow_parameters_for_faiss(feature_run_id, client):
    """Retrieve necessary parameters from MLflow for FAISS indexing."""
    print(f"Retrieving parameters from feature run ID: {feature_run_id}")
    try:
        feature_run = client.get_run(feature_run_id)
        print(f"Successfully retrieved run object for {feature_run_id}")
    except Exception as e:
        print(f"ERROR: Could not retrieve run {feature_run_id} from MLflow.")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Error details: {e}")
        traceback.print_exc()
        raise ValueError(f"Could not retrieve run {feature_run_id}. Error: {e}")

    # Extract parameters
    embeddings_path = feature_run.data.params.get("embeddings_path")
    exp_dir = feature_run.data.params.get("exp_dir") # Directory base esperimento

    if not embeddings_path:
        raise ValueError(f"Could not find 'embeddings_path' parameter in run {feature_run_id}")
    if not exp_dir:
         raise ValueError(f"Could not find 'exp_dir' parameter in run {feature_run_id}")

    print(f"Found embeddings_path: {embeddings_path}")
    print(f"Found exp_dir: {exp_dir}")

    # Recupera anche eventuali parametri precedenti che potrebbero essere utili (opzionale)
    # num_features = feature_run.data.metrics.get("num_points") # Se avevi loggato questa metrica
    # feature_dim = feature_run.data.metrics.get("num_dimensions") # Se avevi loggato questa metrica

    return {
        "embeddings_path": embeddings_path,
        "exp_dir": exp_dir
        # "num_features": num_features, # Esempio
        # "feature_dim": feature_dim    # Esempio
    }

def log_artifact_to_mlflow(local_path, client, run_id, artifact_subdir="faiss_index"):
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
        print(f"  ERROR logging artifact '{artifact_name}' to MLflow run '{run_id}':")
        print(f"  MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"  Error details: {e}")

def main():
    args = parse_arguments()
    print("--- FAISS Index Building Script ---")
    print(f"Arguments: {args}")

    try:
        # --- MLflow Setup ---
        if args.tracking_uri:
            print(f"Setting MLflow tracking URI to: {args.tracking_uri}")
            mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        print(f"MLflow tracking URI active: {mlflow.get_tracking_uri()}")
        print(f"Logging to MLflow enabled: {args.log_to_mlflow}")
        mlflow_client = client if args.log_to_mlflow else None
        target_run_id = args.feature_run_id # Logghiamo nella stessa run delle features

        print("\n--- Retrieving Parameters via MLflow ---")
        mlflow_params = get_mlflow_parameters_for_faiss(args.feature_run_id, client)
        features_path = mlflow_params['embeddings_path']
        exp_dir = mlflow_params['exp_dir']

        # Costruisci il path completo per l'output dell'indice
        output_index_path = os.path.join(exp_dir, args.output_index_filename)
        print(f"Output index will be saved to: {output_index_path}")
        os.makedirs(exp_dir, exist_ok=True) # Assicura che exp_dir esista

        # --- Load Features ---
        print(f"\n--- Loading Features ---")
        print(f"Loading features from: {features_path}")
        t0 = time.time()
        if not os.path.exists(features_path):
             raise FileNotFoundError(f"Features file not found at: {features_path}")
        features = np.load(features_path)
        # Preprocessing per FAISS/cuML
        if features.dtype != np.float32:
            print(f"Warning: Features dtype is {features.dtype}. Converting to float32.")
            features = features.astype(np.float32)
        if not features.flags['C_CONTIGUOUS']:
             print("Warning: Features array is not C-contiguous. Making a copy.")
             features = np.ascontiguousarray(features)
        load_time = time.time() - t0
        print(f"Loaded {features.shape[0]} features with dimension {features.shape[1]} in {load_time:.2f} seconds.")
        if features.shape[0] == 0: raise ValueError("Features file is empty.")

        # --- Normalize (Optional) ---
        if args.normalize:
            print("\n--- Normalizing Features ---")
            t0 = time.time()
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10 # Evita divisione per zero
            features /= norms
            norm_time = time.time() - t0
            print(f"Normalization done in {norm_time:.2f} seconds.")
        else:
            norm_time = 0.0

        # --- Build FAISS Index ---
        print("\n--- Building FAISS Index ---")
        d = features.shape[1]; n = features.shape[0]
        print(f"Index Type: {args.index_type}, Use GPU: {args.use_gpu}")
        if args.index_type == 'IndexFlatL2': cpu_index = faiss.IndexFlatL2(d)
        elif args.index_type == 'IndexFlatIP': cpu_index = faiss.IndexFlatIP(d)
        else: raise ValueError(f"Unsupported index type: {args.index_type}")

        index_to_use = cpu_index
        gpu_res = None
        gpu_transfer_time = 0.0
        if args.use_gpu:
            if faiss.get_num_gpus() > 0:
                print(f"Using GPU {faiss.get_num_gpus()} resources.")
                try:
                    t_gpu0 = time.time()
                    gpu_res = faiss.StandardGpuResources()
                    index_to_use = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                    gpu_transfer_time = time.time() - t_gpu0
                    print(f"Index transferred to GPU in {gpu_transfer_time:.2f}s.")
                except Exception as e:
                    print(f"Warning: Failed to use GPU ({e}). Falling back to CPU.")
                    index_to_use = cpu_index
            else:
                print("Warning: --use_gpu specified, but no GPUs found. Using CPU.")
                index_to_use = cpu_index

        print(f"Adding {n} vectors to the index...")
        t0 = time.time()
        index_to_use.add(features)
        add_time = time.time() - t0
        print(f"Vectors added in {add_time:.2f} seconds.")
        print(f"Index is trained: {index_to_use.is_trained}, Total vectors: {index_to_use.ntotal}")

        # --- Save Index ---
        print("\n--- Saving FAISS Index ---")
        cpu_transfer_time = 0.0
        if args.use_gpu and faiss.get_num_gpus() > 0 and gpu_res is not None:
            try:
                print("Transferring index from GPU to CPU for saving...")
                t_cpu0 = time.time()
                index_to_save = faiss.index_gpu_to_cpu(index_to_use)
                cpu_transfer_time = time.time() - t_cpu0
                print(f"Transfer to CPU done in {cpu_transfer_time:.2f}s.")
            except Exception as e:
                 print(f"Warning: Could not transfer index from GPU ({e}). Saving might fail.")
                 index_to_save = cpu_index # Fallback
        else:
            index_to_save = index_to_use # Already a CPU index

        print(f"Saving index to: {output_index_path}")
        t0 = time.time()
        faiss.write_index(index_to_save, output_index_path)
        save_time = time.time() - t0
        print(f"Index saved in {save_time:.2f} seconds.")

        # --- Log to MLflow (if requested) ---
        if args.log_to_mlflow:
            print("\n--- Logging Results to MLflow ---")
            if mlflow_client and target_run_id:
                # 1. Log l'indice FAISS
                log_artifact_to_mlflow(output_index_path, mlflow_client, target_run_id, artifact_subdir="faiss_index")

                # 2. Log i parametri di costruzione
                faiss_params = {
                    "feature_run_id": args.feature_run_id,
                    "index_type": args.index_type,
                    "normalized_features": args.normalize,
                    "used_gpu": args.use_gpu and (gpu_res is not None),
                    "num_vectors": index_to_save.ntotal,
                    "vector_dimension": index_to_save.d,
                    "saved_index_filename": args.output_index_filename,
                    "index_size_bytes": os.path.getsize(output_index_path) if os.path.exists(output_index_path) else -1
                }
                try:
                    # Log come parametri MLflow (chiavi semplici)
                    mlflow_client.log_param(target_run_id, "faiss_index_type", faiss_params["index_type"])
                    mlflow_client.log_param(target_run_id, "faiss_normalized", faiss_params["normalized_features"])
                    mlflow_client.log_param(target_run_id, "faiss_used_gpu", faiss_params["used_gpu"])
                    # Log come metriche MLflow (valori numerici)
                    mlflow_client.log_metric(target_run_id, "faiss_num_vectors", faiss_params["num_vectors"])
                    mlflow_client.log_metric(target_run_id, "faiss_vector_dim", faiss_params["vector_dimension"])
                    mlflow_client.log_metric(target_run_id, "faiss_index_size_mb", faiss_params["index_size_bytes"] / (1024*1024) if faiss_params["index_size_bytes"] > 0 else 0)
                    # Log i tempi
                    mlflow_client.log_metric(target_run_id, "faiss_time_load_s", load_time)
                    mlflow_client.log_metric(target_run_id, "faiss_time_norm_s", norm_time)
                    mlflow_client.log_metric(target_run_id, "faiss_time_add_s", add_time)
                    mlflow_client.log_metric(target_run_id, "faiss_time_save_s", save_time)
                    if faiss_params["used_gpu"]:
                         mlflow_client.log_metric(target_run_id, "faiss_time_gpu_transfer_s", gpu_transfer_time)
                         mlflow_client.log_metric(target_run_id, "faiss_time_cpu_transfer_s", cpu_transfer_time)

                    # Log il dizionario completo come artifact JSON
                    params_json_path = os.path.join(os.path.dirname(output_index_path), "faiss_build_params.json")
                    with open(params_json_path, 'w') as f:
                        json.dump(faiss_params, f, indent=4)
                    log_artifact_to_mlflow(params_json_path, mlflow_client, target_run_id, artifact_subdir="faiss_index")
                    os.remove(params_json_path) # Rimuovi il file temporaneo dopo il log

                    print("Successfully logged parameters and metrics to MLflow.")
                except Exception as e:
                    print(f"Warning: Failed to log some parameters/metrics to MLflow. Error: {e}")
            else:
                print("Warning: log_to_mlflow specified, but client or run_id is invalid.")

        print("\n--- FAISS Index Building Script Finished Successfully ---")

    except FileNotFoundError as e:
         print(f"FATAL ERROR: Input file not found. {e}")
         traceback.print_exc()
         exit(1)
    except ValueError as e:
         print(f"FATAL ERROR: Invalid value or configuration. {e}")
         traceback.print_exc()
         exit(1)
    except Exception as e:
        print(f"\n--- An Unexpected FATAL Error Occurred During FAISS Build ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        exit(1)
    finally:
        print("\n--- Script End ---")

if __name__ == "__main__":
    main()