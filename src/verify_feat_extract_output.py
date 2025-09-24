import h5py
import numpy as np
import json
import os
import argparse
import pandas as pd
import sys

def run_verification(run_directory):
    """
    Esegue una serie di test su un file HDF5 di feature e sul suo file di metadati.

    Args:
        run_directory (str): Il percorso della cartella dell'esperimento che contiene
                             metadata.json e features_data_fast.h5.
    """
    print("=" * 60)
    print(f"🔬 Avvio verifica per la cartella: {run_directory}")
    print("=" * 60)

    # --- 1. VERIFICA ESISTENZA FILE ---
    print("\n--- [FASE 1/4] Controllo esistenza file...")
    
    metadata_path = os.path.join(run_directory, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"❌ FALLITO: File di metadati non trovato in: {metadata_path}")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✔️  File di metadati caricato.")

    hdf5_path = os.path.join(run_directory, metadata.get("features_filename", "features_data_fast.h5"))
    if not os.path.exists(hdf5_path):
        print(f"❌ FALLITO: File HDF5 non trovato in: {hdf5_path}")
        sys.exit(1)
    print(f"✔️  File HDF5 trovato.")

    # --- 2. VERIFICA STRUTTURALE HDF5 ---
    print("\n--- [FASE 2/4] Controllo struttura file HDF5...")
    try:
        with h5py.File(hdf5_path, 'r') as h5f:
            # Esistenza dei dataset
            assert 'features' in h5f, "Dataset 'features' mancante."
            assert 'image_paths' in h5f, "Dataset 'image_paths' mancante."
            print("✔️  Dataset 'features' e 'image_paths' presenti.")
            
            # Controllo dimensioni
            expected_rows = metadata['feature_count']
            expected_dims = metadata['feature_dimension']
            features_shape = h5f['features'].shape
            paths_shape = h5f['image_paths'].shape

            assert features_shape == (expected_rows, expected_dims), \
                f"Shape 'features' errata. Attesa: {(expected_rows, expected_dims)}, Trovata: {features_shape}"
            print(f"✔️  Shape 'features' corretta: {features_shape}")

            assert paths_shape == (expected_rows,), \
                f"Shape 'image_paths' errata. Attesa: {(expected_rows,)}, Trovata: {paths_shape}"
            print(f"✔️  Shape 'image_paths' corretta: {paths_shape}")

            # Controllo tipo di dati
            assert h5f['features'].dtype == np.float32, f"dtype 'features' errato. Atteso: float32, Trovato: {h5f['features'].dtype}"
            assert h5py.check_string_dtype(h5f['image_paths'].dtype), "dtype 'image_paths' non è di tipo stringa."
            print("✔️  Tipi di dati (dtype) corretti.")

    except Exception as e:
        print(f"❌ FALLITO: Errore durante il controllo strutturale. {e}")
        sys.exit(1)

    # --- 3. CONTROLLO CORRISPONDENZA DATI ---
    print("\n--- [FASE 3/4] Controllo corrispondenza dei dati (Sanity Check)...")
    try:
        # Carica la lista originale dei file per confronto
        source_list_path = os.path.join(
            metadata['merged_cutout_folder_path'], 
            metadata['merged_cutouts_info_filename']
        )
        source_df = pd.read_json(source_list_path, orient="index")
        print(f"✔️  File di origine dei dati caricato per confronto.")
        
        with h5py.File(hdf5_path, 'r') as h5f:
            num_samples = h5f['features'].shape[0]
            # Seleziona alcuni indici da testare: primo, ultimo e tre casuali
            indices_to_test = list(set([0, num_samples - 1] + list(np.random.randint(1, num_samples - 2, size=3))))

            print(f"Campionamento di {len(indices_to_test)} indici per la verifica: {indices_to_test}")
            
            for idx in indices_to_test:
                # Estrai il percorso dal file HDF5 (decodificandolo da bytes a stringa)
                path_from_h5 = h5f['image_paths'][idx].decode('utf-8')
                
                # Estrai il percorso originale dal DataFrame
                path_from_source = source_df.iloc[idx]['file_path']

                assert path_from_h5 == path_from_source, \
                    f"Disallineamento all'indice {idx}. H5: {path_from_h5}, Sorgente: {path_from_source}"
            
            print("✔️  Tutti i campioni testati corrispondono! I dati sono allineati.")

    except Exception as e:
        print(f"❌ FALLITO: Errore durante il controllo di corrispondenza. {e}")
        sys.exit(1)

    # --- 4. CONTROLLO CORRUZIONE FEATURE ---
    print("\n--- [FASE 4/4] Controllo corruzione delle feature (NaN/inf)...")
    try:
        with h5py.File(hdf5_path, 'r') as h5f:
            # Controlla per NaN o Inf su un campione di feature per efficienza
            sample_size = min(1000, h5f['features'].shape[0]) # Controlla max 1000 righe
            feature_sample = h5f['features'][:sample_size]
            
            has_nan = np.isnan(feature_sample).any()
            has_inf = np.isinf(feature_sample).any()

            assert not has_nan, "Trovati valori NaN nelle feature."
            assert not has_inf, "Trovati valori Infinito nelle feature."

            print("✔️  Nessun valore NaN o Infinito trovato nel campione di feature.")
            
    except Exception as e:
        print(f"❌ FALLITO: Errore durante il controllo di corruzione. {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("✅  SUCCESSO! Tutti i test sono stati superati.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica l'integrità di un file HDF5 di feature.")
    parser.add_argument(
        "run_directory", 
        type=str, 
        help="Il percorso della cartella dell'esperimento (es. outputs/features/features_cecconello_ssl_...)"
    )
    args = parser.parse_args()
    
    run_verification(args.run_directory)