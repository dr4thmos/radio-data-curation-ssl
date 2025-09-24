import h5py
import numpy as np
import argparse
import os
import sys
import time

def convert_h5_to_npy(input_h5_path: str):
    """
    Legge il dataset 'features' da un file HDF5 e lo salva in un file .npy.

    Args:
        input_h5_path (str): Percorso completo del file HDF5 di input.
    """
    print("=" * 60)
    print(f"🚀 Avvio conversione da HDF5 a NumPy")
    print(f"File di input: {input_h5_path}")
    print("=" * 60)

    # 1. Verifica che il file di input esista
    if not os.path.exists(input_h5_path):
        print(f"❌ ERRORE: File di input non trovato: {input_h5_path}")
        sys.exit(1)

    # 2. Definisci il percorso di output
    # Il file .npy avrà lo stesso nome e percorso, ma con estensione diversa.
    output_npy_path = os.path.splitext(input_h5_path)[0] + '.npy'
    print(f"File di output: {output_npy_path}")

    try:
        # 3. Leggi i dati HDF5 e caricali in memoria RAM
        print("\nLeggendo il file HDF5... (potrebbe richiedere un po' di tempo)")
        start_time = time.time()
        with h5py.File(input_h5_path, 'r') as h5_file:
            # Verifica che il dataset 'features' esista
            if 'features' not in h5_file:
                print(f"❌ ERRORE: Dataset 'features' non trovato nel file HDF5.")
                sys.exit(1)
            
            # Carica l'intero dataset in un array NumPy in memoria
            data = h5_file['features'][...]
        
        read_time = time.time() - start_time
        print(f"✔️  Dati caricati in memoria in {read_time:.2f} secondi.")
        print(f"Shape dei dati: {data.shape}")
        print(f"Tipo di dati (dtype): {data.dtype}")

        # 4. Salva l'array NumPy su disco
        print("\nSalvataggio del file .npy... (QUESTA OPERAZIONE SARÀ LENTA)")
        start_time = time.time()
        np.save(output_npy_path, data)
        save_time = time.time() - start_time
        print(f"✔️  File .npy salvato con successo in {save_time:.2f} secondi.")

    except Exception as e:
        print(f"❌ ERRORE: Si è verificato un problema durante la conversione.")
        print(e)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Conversione completata con successo!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte il dataset 'features' da un file HDF5 a un file NumPy (.npy)."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Percorso completo del file features_data_fast.h5 da convertire."
    )
    args = parser.parse_args()

    convert_h5_to_npy(args.input_file)