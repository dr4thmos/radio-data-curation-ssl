# plot_umap_physical_neighbors.py
# -*- coding: utf-8 -*-
"""
VISUALIZZAZIONE UMAP CON VICINI FISICI

Scopo:
Questo script verifica se i cutout fisicamente adiacenti (e sovrapposti)
sono anche vicini nello spazio di embedding della UMAP. Lo fa selezionando
alcuni cutout "seme", trovando i loro vicini fisici e disegnando delle
linee di connessione sul grafico UMAP.

Funzionamento:
1.  Carica le coordinate UMAP e i metadati da info.json.
2.  Converte i metadati in un DataFrame Pandas per ricerche efficienti.
3.  Seleziona N cutout "seme" in modo casuale.
4.  Per ogni seme:
    a. Calcola le coordinate dei suoi 8 vicini fisici diretti.
    b. Cerca questi vicini nel DataFrame (stesso mosaico, posizione adiacente).
5.  Genera un grafico UMAP dove:
    - Tutti i punti sono mostrati in grigio.
    - I punti seme sono evidenziati.
    - I loro vicini fisici sono evidenziati con un altro colore.
    - Vengono tracciate delle linee tra i semi e i loro vicini.

Esempio di utilizzo:
python plot_umap_physical_neighbors.py \
    --info-json /path/to/your/info.json \
    --umap-embedding /path/to/your/umap_embedding.npy \
    --output-file outputs/umap_physical_neighbors.png \
    --num-seeds 20
"""
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import random

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot UMAP highlighting physical neighbors.')
    parser.add_argument('--info-json', type=str, required=True,
                        help='Path to the JSON file with ordered metadata.')
    parser.add_argument('--umap-embedding', type=str, required=True,
                        help='Path to the .npy file with the 2D UMAP coordinates.')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the output plot image.')
    parser.add_argument('--num-seeds', type=int, default=15,
                        help='Number of random seed cutouts to analyze.')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction between cutouts (e.g., 0.5 for 50%).')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for the saved image.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print("--- Loading and Preparing Data ---")
    
    # Carica coordinate UMAP
    umap_coords = np.load(args.umap_embedding)

    # Carica info.json e convertilo in un DataFrame
    with open(args.info_json, 'r') as f:
        info_data = json.load(f)

    # Converte il dizionario in una lista di dizionari e poi in un DataFrame
    # Aggiunge l'indice originale che corrisponde alla riga della UMAP
    records = []
    for index, data in info_data.items():
        record = data.copy()
        record['umap_index'] = int(index)
        records.append(record)
    df = pd.DataFrame(records)
    
    # Estrai le coordinate x e y dalla colonna 'position'
    df[['pos_x', 'pos_y']] = pd.DataFrame(df['position'].tolist(), index=df.index)
    
    # Crea un indice multi-livello per ricerche super veloci
    df.set_index(['mosaic_name', 'pos_x', 'pos_y'], inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Loaded and processed {len(df)} records.")

    # Seleziona N semi casuali
    random_indices = random.sample(range(len(df)), args.num_seeds)
    seed_points = df.iloc[random_indices].copy()
    
    print(f"Selected {args.num_seeds} random seed points.")

    # --- Troviamo i Vicini per ogni Seme ---
    connections = []
    all_neighbor_indices = set()
    
    # La distanza tra i centri dei cutout è size * (1 - overlap)
    # Assumiamo che 'size' sia costante, prendiamo il primo valore
    patch_size = seed_points['size'].iloc[0]
    step = int(patch_size * (1.0 - args.overlap))
    
    print(f"Calculated step between cutouts: {step} pixels.")

    for _, seed in seed_points.iterrows():
        seed_index = seed['umap_index']
        mosaic = seed.name[0]
        x, y = seed.name[1], seed.name[2]
        
        # Definisci le posizioni relative degli 8 vicini
        neighbor_offsets = [
            (-step, -step), (-step, 0), (-step, step),
            (0, -step),                 (0, step),
            (step, -step),  (step, 0),  (step, step)
        ]

        seed_neighbors = []
        for dx, dy in neighbor_offsets:
            neighbor_pos = (mosaic, x + dx, y + dy)
            # Cerca il vicino nel DataFrame indicizzato
            if neighbor_pos in df.index:
                neighbor_data = df.loc[neighbor_pos]
                neighbor_index = neighbor_data['umap_index']
                seed_neighbors.append(neighbor_index)
        
        if seed_neighbors:
            connections.append({'seed': seed_index, 'neighbors': seed_neighbors})
            all_neighbor_indices.update(seed_neighbors)

    all_seed_indices = set(seed_points['umap_index'])
    
    print(f"Found connections for {len(connections)} seeds.")

    # --- Genera il Grafico ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(16, 12))

    # 1. Sfondo con tutti i punti
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=1, color='lightgray', alpha=0.3)

    # 2. Evidenzia tutti i semi e i vicini trovati per dar loro contesto
    all_involved_indices = list(all_seed_indices.union(all_neighbor_indices))
    ax.scatter(umap_coords[all_involved_indices, 0], umap_coords[all_involved_indices, 1],
               s=15, color='gray', alpha=0.6, zorder=2)

    # 3. Disegna le connessioni e i punti specifici
    for conn in connections:
        seed_idx = conn['seed']
        neighbor_indices = conn['neighbors']
        
        # Disegna linee dal seme ai vicini
        for neighbor_idx in neighbor_indices:
            ax.plot(
                [umap_coords[seed_idx, 0], umap_coords[neighbor_idx, 0]],
                [umap_coords[seed_idx, 1], umap_coords[neighbor_idx, 1]],
                color='orange',
                linewidth=0.8,
                alpha=0.7,
                zorder=3
            )
    
    # 4. Disegna i punti seme e i vicini sopra le linee
    ax.scatter(umap_coords[list(all_neighbor_indices), 0], umap_coords[list(all_neighbor_indices), 1],
               s=30, color='deepskyblue', label='Physical Neighbors', zorder=4, edgecolor='black', linewidth=0.5)
    ax.scatter(umap_coords[list(all_seed_indices), 0], umap_coords[list(all_seed_indices), 1],
               s=60, color='crimson', label='Seed Points', zorder=5, edgecolor='black', linewidth=0.5)

    ax.set_title(f'UMAP with Physical Neighbor Connections ({args.num_seeds} seeds)', fontsize=18)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    print(f"Saving plot to {args.output_file}...")
    fig.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    
    print("--- Process Completed ---")

if __name__ == '__main__':
    main()