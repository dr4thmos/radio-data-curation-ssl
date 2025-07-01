# plot_umap_by_category.py
# -*- coding: utf-8 -*-
"""
VISUALIZZAZIONE UMAP PER CATEGORIA SINGOLA

Scopo:
Questo script carica una proiezione UMAP e un file di etichette per creare
una serie di visualizzazioni. Per ogni categoria unica presente nelle etichette,
genera un grafico separato che evidenzia solo i punti appartenenti a quella
categoria su uno sfondo di tutti gli altri punti.

Funzionamento:
1.  Carica le coordinate UMAP e i percorsi ordinati da un file info.json.
2.  Carica e processa le etichette da un file CSV.
3.  Gestisce le etichette multiple (es. ['artefact', 'diffuse']) trattando
    ogni etichetta in modo indipendente. Un punto può apparire in più grafici.
4.  Per ogni etichetta unica (es. 'artefact', 'diffuse', 'FR', etc.):
    a. Crea un nuovo grafico.
    b. Disegna tutti i punti del dataset in grigio come sfondo.
    c. Evidenzia i punti che hanno quell'etichetta con un colore acceso.
    d. Salva il grafico in un file con un nome descrittivo.

Esempio di utilizzo:
python plot_umap_by_category.py \
    --info-json /path/to/your/info.json \
    --umap-embedding /path/to/your/umap_embedding.npy \
    --labels-csv /path/to/your/labels.csv \
    --output-dir outputs/umap_by_category/
"""
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a separate UMAP plot for each category.')
    parser.add_argument('--info-json', type=str, required=True,
                        help='Path to the JSON file with ordered metadata and file paths.')
    parser.add_argument('--umap-embedding', type=str, required=True,
                        help='Path to the .npy file with the 2D UMAP coordinates.')
    parser.add_argument('--labels-csv', type=str, required=True,
                        help='Path to the CSV file with file paths and categories.')
    # --- MODIFICA: Da --output-file a --output-dir ---
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the output plot images.')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for the saved images (lower for faster generation).')
    return parser.parse_args()

def parse_category_string(cat_string):
    """Safely parses a string that looks like a list into a Python list."""
    if not isinstance(cat_string, str): return []
    try:
        parsed_list = ast.literal_eval(cat_string)
        return parsed_list if isinstance(parsed_list, list) else []
    except (ValueError, SyntaxError): return []

def main():
    args = parse_arguments()

    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)

    # --- (Caricamento dati: UMAP, info.json, labels.csv - invariato) ---
    print("--- Loading Data ---")
    umap_coords = np.load(args.umap_embedding)
    
    with open(args.info_json, 'r') as f:
        info_data = json.load(f)
    all_paths = [v['file_path'].strip() for k, v in sorted(info_data.items(), key=lambda i: int(i[0]))]

    labels_df = pd.read_csv(args.labels_csv)
    labels_df['file_path'] = labels_df['file_path'].str.strip()
    labels_df['categories'] = labels_df['categories'].apply(parse_category_string)
    
    print(f"Loaded {umap_coords.shape[0]} UMAP points and {len(labels_df)} labels.")

    # --- (Mappatura: percorso -> indice - invariato) ---
    print("\n--- Mapping Labels to UMAP Coordinates ---")
    path_to_index_map = {path: i for i, path in enumerate(all_paths)}
    labels_df['umap_index'] = labels_df['file_path'].map(path_to_index_map)
    labeled_data = labels_df.dropna(subset=['umap_index']).copy()
    labeled_data['umap_index'] = labeled_data['umap_index'].astype(int)
    print(f"Successfully mapped {len(labeled_data)} entries.")

    # --- MODIFICA CHIAVE: Gestione delle etichette multiple ---
    # Esplodiamo il DataFrame: se una riga ha ['artefact', 'diffuse'],
    # diventerà due righe, una per 'artefact' e una per 'diffuse'.
    # Questo semplifica enormemente l'iterazione successiva.
    exploded_labels = labeled_data.explode('categories').rename(columns={'categories': 'category'})
    
    # Ottieni la lista di tutte le categorie uniche
    all_unique_categories = sorted(exploded_labels['category'].unique())
    
    print(f"\nFound {len(all_unique_categories)} unique categories to plot:")
    print(all_unique_categories)

    # --- NUOVA LOGICA: Ciclo di plotting per ogni categoria ---
    print("\n--- Generating Plots for Each Category ---")
    
    for i, category in enumerate(all_unique_categories):
        print(f"  ({i+1}/{len(all_unique_categories)}) Plotting category: '{category}'...")

        # Seleziona i dati solo per la categoria corrente
        category_df = exploded_labels[exploded_labels['category'] == category]
        highlight_indices = category_df['umap_index'].values
        num_points_in_category = len(highlight_indices)

        # Crea la figura
        fig, ax = plt.subplots(figsize=(14, 10))

        # 1. Disegna lo sfondo con tutti i punti
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            s=2,
            color='lightgray',
            alpha=0.4
        )

        # 2. Evidenzia i punti di questa categoria
        ax.scatter(
            umap_coords[highlight_indices, 0],
            umap_coords[highlight_indices, 1],
            s=40,
            color='crimson', # Un colore brillante e fisso per l'evidenziazione
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Imposta titoli e etichette
        title = f"UMAP Projection: Highlighting '{category}' ({num_points_in_category} points)"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Salva il file
        # Pulisci il nome della categoria per renderlo un nome di file valido
        safe_category_name = "".join(c for c in category if c.isalnum() or c in (' ', '_')).rstrip()
        output_filename = f"umap_plot_category_{safe_category_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig) # Chiudi la figura per liberare memoria

    print("\n--- All Plots Generated Successfully ---")

if __name__ == '__main__':
    main()