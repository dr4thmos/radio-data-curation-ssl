import json
import os

def filter_overlapping_cutouts_keep_indices(input_json_path, output_json_path):
    """
    Filtra un file JSON che descrive cutouts generati con overlap,
    mantenendo solo quelli che corrisponderebbero a una griglia non sovrapposta.
    Gli indici originali vengono conservati nel file di output.

    Args:
        input_json_path (str): Percorso del file JSON di input con i cutouts sovrapposti.
        output_json_path (str): Percorso in cui verrà salvato il nuovo file JSON senza overlap.
    """
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file di input '{input_json_path}' non trovato.")
        return
    except json.JSONDecodeError:
        print(f"Errore: Impossibile decodificare il file JSON '{input_json_path}'.")
        return

    if not data:
        print("Il file JSON di input è vuoto. Nessun dato da processare.")
        with open(output_json_path, 'w') as f:
            json.dump({}, f, indent=4)
        print(f"Creato un file di output vuoto in '{output_json_path}'.")
        return

    # Ordina le chiavi per assicurarsi di partire dal primo cutout "logico",
    # sebbene per la tua struttura con chiavi numeriche non sia strettamente necessario.
    sorted_keys = sorted(data.keys(), key=int)
    
    first_key = sorted_keys[0]
    size = data[first_key]['size']
    start_row, start_col = data[first_key]['position']

    filtered_data = {}

    print(f"Processing {len(data)} cutouts. Mantenendo gli indici originali.")
    print(f"Dimensione della finestra: {size}")
    print(f"Posizione di riferimento (dal primo cutout): [{start_row}, {start_col}]")

    # Itera su tutti gli elementi del dizionario originale
    for key, entry in data.items():
        current_row, current_col = entry['position']

        # Calcola la differenza rispetto alla posizione iniziale
        delta_row = current_row - start_row
        delta_col = current_col - start_col

        # Controlla se le differenze sono multipli esatti della dimensione.
        # Questo identifica i cutouts che sarebbero stati generati con step = size.
        if delta_row % size == 0 and delta_col % size == 0:
            # Se la condizione è soddisfatta, aggiungi l'elemento al nuovo dizionario
            # USANDO LA CHIAVE ORIGINALE.
            filtered_data[key] = entry

    print(f"Filtrati {len(filtered_data)} cutouts.")

    try:
        with open(output_json_path, 'w') as f:
            json.dump(filtered_data, f, indent=4) # indent=4 per una migliore leggibilità
        print(f"Dati filtrati salvati in '{output_json_path}'.")
    except IOError:
        print(f"Errore: Impossibile scrivere il file di output '{output_json_path}'.")

# --- Esempio di utilizzo ---

# Crea un file JSON di esempio per il test
sample_data = {
    "0": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_128_3072.npy",
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [128, 3072],
        "size": 256
    },
    "1": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_128_3200.npy", # Overlap with 0
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [128, 3200],
        "size": 256
    },
    "2": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_128_3328.npy", # Non-overlapping with 0 horizontally
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [128, 3328], # 3328 = 3072 + 256 (step = size)
        "size": 256
    },
    "3": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_128_3456.npy", # Overlap
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [128, 3456], # 3456 = 3328 + 128 (step = size/2)
        "size": 256
    },
     "4": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_256_3072.npy", # Overlap vertically
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [256, 3072], # 256 = 128 + 128 (step = size/2)
        "size": 256
    },
     "5": {
        "file_path": "outputs/cutouts/cutouts-strategy_sliding_window-overlap_05-size_256_30d67552/P184+57/npy/patch_384_3072.npy", # Non-overlapping vertically
        "survey": "LoTTS",
        "mosaic_name": "P184+57",
        "position": [384, 3072], # 384 = 128 + 256 (step = size)
        "size": 256
    }
}

input_file = "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/merged_cutouts/merged_cutouts_74003e44/info.json"
output_file = "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/merged_cutouts/merged_cutouts_74003e44/info_non_overlapped.json"


"""
# Salva i dati di esempio nel file di input
with open(input_file, 'w') as f:
    json.dump(sample_data, f, indent=4)
"""
# Esegui la funzione di filtraggio
filter_overlapping_cutouts_keep_indices(input_file, output_file)