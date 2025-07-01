# analyze_feature_distances.py
# -*- coding: utf-8 -*-
"""
ANALISI STATISTICA DELLE DISTANZE TRA FEATURES

Scopo:
Questo script valuta quantitativamente se i cutout fisicamente adiacenti
sono più vicini nello spazio delle features ad alta dimensione rispetto a
coppie di cutout casuali.

Funzionamento:
1.  Carica le features ad alta dimensione da un file HDF5 e i metadati da info.json.
2.  Campiona N coppie di cutout fisicamente adiacenti.
3.  Campiona N coppie di cutout scelti casualmente (gruppo di controllo).
4.  Per ogni coppia, calcola la distanza euclidea tra i loro vettori di features.
5.  Genera due distribuzioni di distanze: "vicini fisici" e "casuali".
6.  Visualizza le distribuzioni con un grafico di densità (KDE plot).
7.  Calcola statistiche descrittive (media, mediana) e un test di significatività
    (Mann-Whitney U) per confrontare le due distribuzioni.

Esempio di utilizzo:
python analyze_feature_distances.py \
    --h5-file /path/to/features_data.h5 \
    --info-json /path/to/info.json \
    --output-plot outputs/distance_analysis.png \
    --num-samples 5000
"""
import argparse
import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze feature space distances of physical neighbors.')
    parser.add_argument('--h5-file', type=str, required=True,
                        help='Path to the HDF5 file containing the high-dimensional features.')
    parser.add_argument('--info-json', type=str, required=True,
                        help='Path to the JSON file with ordered metadata.')
    parser.add_argument('--output-plot', type=str, required=True,
                        help='Path to save the output plot image.')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of pairs (both neighbor and random) to sample for the analysis.')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction between cutouts.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for the saved image.')
    return parser.parse_args()

def get_physical_neighbor_pairs(df, num_samples, overlap):
    """Sample pairs of physically adjacent cutouts."""
    neighbor_pairs = []
    
    patch_size = df['size'].iloc[0]
    step = int(patch_size * (1.0 - overlap))
    
    # Pre-calcola gli offset dei vicini
    neighbor_offsets = [
        (-step, 0), (step, 0), (0, -step), (0, step),
        (-step, -step), (-step, step), (step, -step), (step, step)
    ]
    
    # Tenta di trovare coppie fino a raggiungere il numero desiderato
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(neighbor_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        # Scegli un punto seme casuale
        seed = df.sample(1).iloc[0]
        seed_index = seed['umap_index']
        mosaic, x, y = seed.name
        
        # Scegli un offset di vicino casuale
        dx, dy = random.choice(neighbor_offsets)
        neighbor_pos = (mosaic, x + dx, y + dy)
        
        if neighbor_pos in df.index:
            neighbor_data = df.loc[neighbor_pos]
            neighbor_index = neighbor_data['umap_index']
            # Aggiungi la coppia (indice1, indice2)
            neighbor_pairs.append(tuple(sorted((seed_index, neighbor_index))))
            
    # Rimuovi eventuali duplicati
    return list(set(neighbor_pairs))

def get_random_pairs(df, num_samples):
    """Sample pairs of random, non-adjacent cutouts."""
    random_pairs = []
    all_indices = df['umap_index'].values
    
    while len(random_pairs) < num_samples:
        idx1, idx2 = np.random.choice(all_indices, 2, replace=False)
        random_pairs.append(tuple(sorted((idx1, idx2))))
        
    return list(set(random_pairs))

def main():
    args = parse_arguments()

    print("--- Loading and Preparing Data ---")
    # Carica metadati e crea DataFrame indicizzato
    with open(args.info_json, 'r') as f:
        info_data = json.load(f)
    records = [{'umap_index': int(k), **v} for k, v in info_data.items()]
    df = pd.DataFrame(records)
    df[['pos_x', 'pos_y']] = pd.DataFrame(df['position'].tolist(), index=df.index)
    df.set_index(['mosaic_name', 'pos_x', 'pos_y'], inplace=True)
    df.sort_index(inplace=True)

    print("--- Sampling Pairs ---")
    # Campiona coppie di vicini e casuali
    neighbor_pairs = get_physical_neighbor_pairs(df, args.num_samples, args.overlap)
    # Assicurati che le coppie casuali siano dello stesso numero
    random_pairs = get_random_pairs(df, len(neighbor_pairs))
    print(f"Sampled {len(neighbor_pairs)} neighbor pairs and {len(random_pairs)} random pairs.")

    print("--- Calculating Distances from HDF5 Features ---")
    neighbor_distances = []
    random_distances = []
    
    with h5py.File(args.h5_file, 'r') as h5f:
        features_dset = h5f['features']
        
        # Calcola distanze per i vicini
        for idx1, idx2 in neighbor_pairs:
            feat1 = features_dset[idx1]
            feat2 = features_dset[idx2]
            dist = euclidean(feat1, feat2)
            neighbor_distances.append(dist)
            
        # Calcola distanze per le coppie casuali
        for idx1, idx2 in random_pairs:
            feat1 = features_dset[idx1]
            feat2 = features_dset[idx2]
            dist = euclidean(feat1, feat2)
            random_distances.append(dist)

    print("--- Analyzing and Plotting Results ---")
    # Crea un DataFrame per il plotting
    dist_df_neighbors = pd.DataFrame({'distance': neighbor_distances, 'type': 'Physical Neighbors'})
    dist_df_random = pd.DataFrame({'distance': random_distances, 'type': 'Random Pairs'})
    plot_df = pd.concat([dist_df_neighbors, dist_df_random])

    # Plot
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=plot_df, x='distance', hue='type', fill=True, common_norm=False)
    plt.title('Distribution of Feature Distances', fontsize=16)
    plt.xlabel('Euclidean Distance in High-Dimensional Feature Space', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Stampa statistiche sul grafico
    mean_neighbor = np.mean(neighbor_distances)
    median_neighbor = np.median(neighbor_distances)
    mean_random = np.mean(random_distances)
    median_random = np.median(random_distances)
    
    stats_text = (
        f"Neighbors: Mean={mean_neighbor:.2f}, Median={median_neighbor:.2f}\n"
        f"Random:    Mean={mean_random:.2f}, Median={median_random:.2f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.savefig(args.output_plot, dpi=args.dpi, bbox_inches='tight')
    print(f"Plot saved to {args.output_plot}")
    plt.close()

    # Esegui test statistico
    # Usiamo Mann-Whitney U perché non assumiamo che le distribuzioni siano normali
    statistic, p_value = mannwhitneyu(neighbor_distances, random_distances, alternative='less')
    
    print("\n--- Statistical Analysis ---")
    print(f"Mean distance (Neighbors): {mean_neighbor:.4f}")
    print(f"Mean distance (Random):   {mean_random:.4f}")
    print("\nMann-Whitney U Test (one-sided: are neighbor distances smaller?):")
    print(f"  Statistic: {statistic:.2f}")
    print(f"  p-value:   {p_value:.10f}")

    if p_value < 0.001:
        print("\nConclusion: The p-value is extremely small.")
        print("This provides strong evidence that the distances between physically adjacent cutouts are")
        print("statistically significantly smaller than for random pairs. The model is performing well in this regard.")
    else:
        print("\nConclusion: The p-value is not small enough to reject the null hypothesis.")
        print("There is no strong statistical evidence that physically adjacent cutouts are closer in feature space.")

if __name__ == '__main__':
    main()