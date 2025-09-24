import json
import numpy as np
import random
import argparse
import os

def sottocampiona_e_salva(json_path, output_path, num_samples):
    """
    Carica un file JSON, sottocampiona un numero specificato di indici (le chiavi del JSON),
    e salva il risultato in un file NumPy (.npy).

    Args:
        json_path (str): Il percorso del file JSON di input.
        output_path (str): Il percorso del file .npy di output.
        num_samples (int): Il numero di indici da sottocampionare.
    """
    # --- 1. Controllo dei file di input e output ---
    if not os.path.exists(json_path):
        print(f"Errore: Il file di input '{json_path}' non è stato trovato.")
        return

    print(f"Caricamento del file JSON da: {json_path}")

    # --- 2. Caricamento del file JSON ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Errore: Il file '{json_path}' non è un JSON valido.")
        return
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")
        return
        
    # --- 3. Estrazione e conversione degli indici ---
    # Le chiavi del JSON sono gli indici, ma sono stringhe. Le convertiamo in interi.
    tutti_gli_indici = [int(key) for key in data.keys()]
    totale_indici = len(tutti_gli_indici)
    print(f"Trovati {totale_indici} indici totali nel file.")

    # --- 4. Controllo del numero di campioni ---
    # Se chiedi più campioni di quelli disponibili, usiamo tutti gli indici.
    if num_samples > totale_indici:
        print(f"Attenzione: Hai richiesto {num_samples} campioni, ma ne sono disponibili solo {totale_indici}.")
        print("Verranno utilizzati tutti gli indici disponibili.")
        num_samples = totale_indici

    # --- 5. Sottocampionamento casuale ---
    # random.sample estrae 'num_samples' elementi unici dalla lista, senza ripetizioni.
    print(f"Sottocampionamento di {num_samples} indici...")
    indici_campionati = random.sample(tutti_gli_indici, num_samples)

    # --- 6. Creazione e salvataggio dell'array NumPy ---
    # Convertiamo la lista di indici campionati in un array NumPy.
    array_numpy = np.array(indici_campionati, dtype=np.int64)

    # Assicuriamoci che la cartella di output esista
    output_dir = os.path.dirname(output_path)
    if output_dir: # Se il percorso contiene una cartella
        os.makedirs(output_dir, exist_ok=True)

    print(f"Salvataggio degli indici campionati in: {output_path}")
    np.save(output_path, array_numpy)

    print("\nOperazione completata con successo!")
    print(f"Creato file '{output_path}' con {len(array_numpy)} indici.")
    print(f"Esempio dei primi 10 indici salvati: {array_numpy[:10]}")


if __name__ == "__main__":
    # --- Configurazione del parser per gli argomenti da riga di comando ---
    parser = argparse.ArgumentParser(
        description="Genera un file NumPy di indici sottocampionati da una lista JSON."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Percorso del file JSON di input."
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Percorso per il file NumPy (.npy) di output."
    )

    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        required=True,
        help="Numero di indici da sottocampionare (es. 300000)."
    )

    args = parser.parse_args()

    # --- Esecuzione della funzione principale con gli argomenti forniti ---
    sottocampiona_e_salva(args.input, args.output, args.num_samples)

    #python random_sampling.py --input /leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/merged_cutouts/merged_cutouts_74003e44/info.json --output outputs/rand_sample1_300k.npy --num_samples 300000