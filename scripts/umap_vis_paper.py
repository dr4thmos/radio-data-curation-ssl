# -*- coding: utf-8 -*-
"""
VISUALIZZAZIONE UMAP AVANZATA CON MARKER A TORTA PER MULTI-LABEL

Scopo:
Esegue un'analisi UMAP completa e multi-sfaccettata su un dataset
semi-supervisionato, gestendo input in formati diversi e sfruttando l'accelerazione GPU.
Include opzioni avanzate come pre-processing con PCA, metrica coseno e modalità
densMAP per una migliore rivelazione della struttura dei dati.

Workflow:
1.  Carica i dati non etichettati (da .npy o .h5) e i dati etichettati (da .h5).
2.  Verifica la coerenza dei dati etichettati con un file info.json ed estrae
    sia le etichette principali che quelle morfologiche.
3.  Opzionalmente, applica StandardScaler e PCA per pre-elaborare i dati.
4.  Esegue tre analisi UMAP separate e indipendenti:
    a) DATI COMBINATI: Mostra la posizione dei dati etichettati rispetto alla
       distribuzione globale.
    b) DATI NON ETICHETTATI: Rivela la struttura intrinseca del dataset principale.
    c) DATI ETICHETTATI: Mostra come le classi si raggruppano tra loro.
5.  Per ogni analisi, i risultati vengono visualizzati generando 5 grafici totali
    e salvati in sottodirectory dedicate.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import h5py
import cupy as cp
from cuml.manifold.umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D

def parse_arguments():
    """Parse command line arguments using direct file paths."""
    parser = argparse.ArgumentParser(description='Advanced UMAP visualization with pie chart markers.')
    
    # Input dati
    parser.add_argument('--unlabeled_features_path', type=str, required=True,
                        help='Path to UNLABELED features file (.h5 or .npy).')
    parser.add_argument('--labeled_features_path', type=str, required=True,
                        help='Path to LABELED features file (.h5, must contain features and image_paths).')
    parser.add_argument('--labeled_info_json_path', type=str, required=True,
                        help='Path to the info.json file for labels.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory to save the output plots and embeddings.')

    # Configurazione Pre-processing
    parser.add_argument('--pca_components', type=int, default=0,
                        help='Number of PCA components to use for pre-processing. '
                             'Set to 0 to disable PCA (default: 0). Recommended: 64, 128.')

    # Configurazione UMAP
    parser.add_argument('--n_neighbors', type=int, default=50, help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist')
    parser.add_argument('--metric', type=str, default='cosine', 
                        help='UMAP distance metric (default: cosine for high-dim data).')
    parser.add_argument('--densmap', action='store_true',
                        help='Enable densMAP to better preserve density information. May be slower.')
    parser.add_argument('--marker_linewidth', type=float, default=0.6,
                        help="Thickness of the pie marker's outer border (in points).")

    # Configurazione Plot
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved images')
    parser.add_argument('--marker_radius', type=float, default=0.1,
                        help='Fixed, absolute radius for the pie chart markers in plot coordinates.')
    parser.add_argument('--zoom_quantile', type=float, default=0.995,
                        help="Quantile to use for auto-zooming on the main data cluster. "
                             "E.g., 0.995 ignores the top/bottom 0.5%% of outliers. "
                             "Set to 1.0 to disable zoom.")

    return parser.parse_args()


def _calculate_zoom_limits(points, quantile_level=0.995, padding_factor=0.1):
    """Calculates x and y axis limits based on data quantiles to zoom on the core distribution."""
    if points.size == 0 or quantile_level >= 1.0:
        return None, None # No zoom needed

    # Calcola i limiti inferiore e superiore per ogni asse basandosi sui quantili
    lower_quantile = (1.0 - quantile_level) / 2.0
    upper_quantile = 1.0 - lower_quantile
    
    xlim_min, xlim_max = np.quantile(points[:, 0], [lower_quantile, upper_quantile])
    ylim_min, ylim_max = np.quantile(points[:, 1], [lower_quantile, upper_quantile])
    
    # Aggiungi un piccolo padding per non avere punti sui bordi
    x_range = xlim_max - xlim_min
    y_range = ylim_max - ylim_min
    
    xlim = (xlim_min - x_range * padding_factor, xlim_max + x_range * padding_factor)
    ylim = (ylim_min - y_range * padding_factor, ylim_max + y_range * padding_factor)
    
    return xlim, ylim

def load_data(features_path, load_paths=False):
    """Loads features and optionally image paths from either an HDF5 or NumPy file."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading data from: {features_path}")
    
    if features_path.endswith(('.h5', '.hdf5')):
        with h5py.File(features_path, 'r') as h5f:
            if 'features' not in h5f: raise KeyError(f"'features' not found in: {features_path}")
            features = h5f['features'][:]
            image_paths = None
            if load_paths:
                if 'image_paths' not in h5f: raise KeyError(f"'image_paths' not found in: {features_path}")
                image_paths = h5f['image_paths'][:]
        return features, image_paths

    elif features_path.endswith('.npy'):
        features = np.load(features_path, mmap_mode='r')
        if load_paths: print(f"Warning: 'load_paths' requested, but input is a .npy file.")
        return features, None
    else:
        raise ValueError(f"Unsupported file format: {features_path}. Use .h5, .hdf5, or .npy.")

def verify_and_extract_labels(image_paths_h5, info_json_path, label_field='label'):
    """Estrae le etichette da un campo specifico (es. 'label' o 'label_morph')."""
    with open(info_json_path, 'r') as f:
        json_data = json.load(f)
    
    labels_list = []
    for i, path_bytes in enumerate(image_paths_h5):
        path_h5 = path_bytes.decode('utf-8')
        path_json = json_data[str(i)]['file_path']
        if os.path.basename(path_h5) != os.path.basename(path_json):
            raise ValueError(f"Mismatch at index {i}: H5:'{os.path.basename(path_h5)}' != JSON:'{os.path.basename(path_json)}'")
        
        labels = sorted(json_data[str(i)].get(label_field, []))
        labels_list.append(tuple(labels))
    return labels_list

def create_pie_marker_plot(title, output_path, dpi, assigned_labels, embedding_labeled, 
                           embedding_unlabeled=None, marker_radius=0.1, marker_linewidth=0.6):
    """Crea e salva un plot UMAP con marker a torta per i dati etichettati."""
    print(f"Generating pie marker plot: {os.path.basename(output_path)}...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 16))

    if embedding_unlabeled is not None:
        ax.scatter(embedding_unlabeled[:, 0], embedding_unlabeled[:, 1], s=1, color='dimgray', alpha=0.2)

    print(f"  - Using fixed marker radius: {marker_radius}")

    all_base_labels = sorted(list(set(label for combo in assigned_labels for label in combo)))
    cmap = plt.colormaps.get('tab20')
    base_color_map = {label: mcolors.to_hex(cmap.colors[i % len(cmap.colors)]) for i, label in enumerate(all_base_labels)}

    marker_alpha = 0.9
    for i in range(len(embedding_labeled)):
        x, y = embedding_labeled[i]
        combo = assigned_labels[i]
        
        if not combo: continue
        
        num_labels = len(combo)
        angle_slice = 360.0 / num_labels
        
        outer_circle = plt.Circle((x, y), marker_radius, facecolor='none', edgecolor='lightgray', linewidth=marker_linewidth, alpha=marker_alpha)
        ax.add_patch(outer_circle)
        
        for j, label in enumerate(combo):
            start_angle = j * angle_slice
            end_angle = (j + 1) * angle_slice
            color = base_color_map.get(label, 'white')
            wedge = Wedge(center=(x, y), r=marker_radius, theta1=start_angle, theta2=end_angle,
                          facecolor=color, edgecolor=None, linewidth=0, alpha=marker_alpha)
            ax.add_patch(wedge)

    all_points = np.vstack([embedding_labeled, embedding_unlabeled]) if embedding_unlabeled is not None else embedding_labeled
    xlim, ylim = _calculate_zoom_limits(all_points, zoom_quantile)
    
    if xlim and ylim:
        print(f"  - Applying zoom to range: X={xlim}, Y={ylim}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.autoscale_view() # Comportamento di default se lo zoom è disabilitato

    ax.set_title(title, fontsize=20, color='white')
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    ax.tick_params(colors='gray')
    ax.spines['bottom'].set_color('gray'); ax.spines['top'].set_color('gray') 
    ax.spines['right'].set_color('gray'); ax.spines['left'].set_color('gray')

    legend_elements = [Line2D([0], [0], marker='o', color='none', label=label,
                              markerfacecolor=color, markersize=12)
                       for label, color in base_color_map.items()]
    ax.legend(handles=legend_elements, title="Legenda Etichette", loc="upper right", fontsize=10)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

def main():
    args = parse_arguments()
    try:
        # --- 0. PREPARA I PARAMETRI ---
        params_suffix = f"nneigh{args.n_neighbors}_mindist{str(args.min_dist).replace('.', 'p')}_metric-{args.metric}"
        if args.pca_components > 0:
            params_suffix += f"_pca{args.pca_components}"
        if args.densmap:
            params_suffix += "_densmap"
        
        # --- 1. CARICA I DATI E LE ETICHETTE ---
        print("\n--- 1. Loading Data & Extracting All Labels ---")
        unlabeled_features, _ = load_data(args.unlabeled_features_path)
        labeled_features, labeled_image_paths = load_data(args.labeled_features_path, load_paths=True)
        
        main_labels = verify_and_extract_labels(labeled_image_paths, args.labeled_info_json_path, label_field='label')
        morph_labels = verify_and_extract_labels(labeled_image_paths, args.labeled_info_json_path, label_field='label_morph')
        
        umap_params = {
            'n_neighbors': args.n_neighbors, 'min_dist': args.min_dist,
            'metric': args.metric, 'densmap': args.densmap, 'verbose': True
        }
        if args.metric == 'cosine':
            umap_params['algorithm'] = 'ivf_flat'
        
        print(f"\nUsing UMAP parameters: {umap_params}")
        if args.pca_components > 0:
            print(f"Using PCA pre-processing with {args.pca_components} components.")

        # --- 2. CALCOLA LE PROIEZIONI UMAP ---
        print("\n--- 2. Calculating UMAP Projections ---")
        
        # Analisi Combinata
        print("\nRunning UMAP on COMBINED data...")
        all_features = np.vstack([unlabeled_features, labeled_features])
        all_features_scaled = StandardScaler().fit_transform(all_features.astype(np.float32))
        
        if args.pca_components > 0:
            pca = PCA(n_components=args.pca_components)
            features_to_embed = pca.fit_transform(all_features_scaled)
            print(f"  PCA applied. Shape after PCA: {features_to_embed.shape}")
        else:
            features_to_embed = all_features_scaled
            
        all_features_gpu = cp.array(features_to_embed)
        umap_combined = UMAP(**umap_params)
        embedding_combined = cp.asnumpy(umap_combined.fit_transform(all_features_gpu))
        embedding_unlabeled_coords = embedding_combined[:len(unlabeled_features)]
        embedding_labeled_coords = embedding_combined[len(unlabeled_features):]
        del all_features, all_features_scaled, features_to_embed, all_features_gpu, umap_combined

        # Analisi solo Non Etichettati
        print("\nRunning UMAP on UNLABELED data...")
        unlabeled_scaled = StandardScaler().fit_transform(unlabeled_features.astype(np.float32))
        
        if args.pca_components > 0:
            pca = PCA(n_components=args.pca_components)
            features_to_embed = pca.fit_transform(unlabeled_scaled)
            print(f"  PCA applied. Shape after PCA: {features_to_embed.shape}")
        else:
            features_to_embed = unlabeled_scaled
            
        unlabeled_gpu = cp.array(features_to_embed)
        umap_unlabeled = UMAP(**umap_params)
        embedding_unlabeled_only = cp.asnumpy(umap_unlabeled.fit_transform(unlabeled_gpu))
        del unlabeled_scaled, features_to_embed, unlabeled_gpu, umap_unlabeled
        
        # Analisi solo Etichettati
        print("\nRunning UMAP on LABELED data...")
        labeled_scaled = StandardScaler().fit_transform(labeled_features.astype(np.float32))
        
        if args.pca_components > 0:
            n_components = min(args.pca_components, len(labeled_scaled))
            if n_components < args.pca_components:
                print(f"  Warning: PCA components reduced to {n_components} due to small sample size.")
            pca = PCA(n_components=n_components)
            features_to_embed = pca.fit_transform(labeled_scaled)
            print(f"  PCA applied. Shape after PCA: {features_to_embed.shape}")
        else:
            features_to_embed = labeled_scaled

        labeled_gpu = cp.array(features_to_embed)
        umap_labeled = UMAP(**umap_params)
        embedding_labeled_only = cp.asnumpy(umap_labeled.fit_transform(labeled_gpu))
        del labeled_scaled, features_to_embed, labeled_gpu, umap_labeled

        # --- 3. CREA LE DIRECTORY DI OUTPUT ---
        run_output_dir = os.path.join(args.output_dir, f"umap_run_{params_suffix}")
        combined_dir = os.path.join(run_output_dir, "combined")
        labeled_only_dir = os.path.join(run_output_dir, "labeled_only")
        unlabeled_only_dir = os.path.join(run_output_dir, "unlabeled_only")
        
        os.makedirs(combined_dir, exist_ok=True)
        os.makedirs(labeled_only_dir, exist_ok=True)
        os.makedirs(unlabeled_only_dir, exist_ok=True)
        print(f"\nSaving results to: {run_output_dir}")

        # --- 4. GENERA TUTTI I PLOT ---
        print("\n--- 4. Generating All Plots ---")
        
        plot_params = {
            'dpi': args.dpi,
            'marker_radius': args.marker_radius,
            'marker_linewidth': args.marker_linewidth,
            'zoom_quantile': args.zoom_quantile
        }

        create_pie_marker_plot(
            title='UMAP Combinata - Etichette Principali',
            output_path=os.path.join(combined_dir, f'combined_main_labels_{params_suffix}.png'),
            assigned_labels=main_labels,
            embedding_labeled=embedding_labeled_coords, embedding_unlabeled=embedding_unlabeled_coords,
            **plot_params
        )
        create_pie_marker_plot(
            title='UMAP Solo Dati Etichettati - Etichette Principali',
            output_path=os.path.join(labeled_only_dir, f'labeled_main_labels_{params_suffix}.png'),
            assigned_labels=main_labels, embedding_labeled=embedding_labeled_only,
            **plot_params
        )
        create_pie_marker_plot(
            title='UMAP Combinata - Etichette Morfologiche',
            output_path=os.path.join(combined_dir, f'combined_morph_labels_{params_suffix}.png'),
            assigned_labels=morph_labels,
            embedding_labeled=embedding_labeled_coords, embedding_unlabeled=embedding_unlabeled_coords,
            **plot_params
        )
        create_pie_marker_plot(
            title='UMAP Solo Dati Etichettati - Etichette Morfologiche',
            output_path=os.path.join(labeled_only_dir, f'labeled_morph_labels_{params_suffix}.png'),
            assigned_labels=morph_labels, embedding_labeled=embedding_labeled_only,
            **plot_params
        )

        print("Generating unlabeled-only plot...")
        plt.style.use('dark_background')
        fig_unlabeled, ax = plt.subplots(figsize=(16, 12))
        ax.scatter(embedding_unlabeled_only[:, 0], embedding_unlabeled_only[:, 1], s=1, alpha=0.3, c='cyan')

        xlim, ylim = _calculate_zoom_limits(embedding_unlabeled_only, args.zoom_quantile)
        if xlim and ylim:
            print(f"  - Applying zoom to range: X={xlim}, Y={ylim}")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_title(f'UMAP: Unlabeled Data Only (n={len(unlabeled_features):,})', fontsize=18, color='white')
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)
        ax.tick_params(colors='gray')
        ax.spines['bottom'].set_color('gray'); ax.spines['top'].set_color('gray')
        ax.spines['right'].set_color('gray'); ax.spines['left'].set_color('gray')
        plt.tight_layout()
        unlabeled_path = os.path.join(unlabeled_only_dir, f'unlabeled_plot_{params_suffix}.png')
        
        fig_unlabeled.savefig(unlabeled_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig_unlabeled)
        print(f"Plot saved to: {unlabeled_path}")

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()