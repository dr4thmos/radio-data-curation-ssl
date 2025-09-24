# -*- coding: utf-8 -*-
"""
SCRIPT DIAGNOSTICO PER CONFRONTARE DUE SET DI FEATURES

Scopo:
Questo script esegue una serie di test quantitativi e visivi per determinare
se due set di features ad alta dimensionalità provengono da distribuzioni simili.
È utile quando le visualizzazioni UMAP mostrano comportamenti inattesi.

Workflow:
1.  Carica i due set di features (uno grande, uno piccolo).
2.  Sottocampiona il set grande per rendere i confronti equi.
3.  Esegue 4 analisi:
    a) STATISTICHE DI BASE: Confronta media e deviazione standard globali.
    b) DISTRIBUZIONI 1D: Plotta le densità (KDE) di alcune features casuali.
    c) PROIEZIONE PCA: Addestra PCA sul set grande e proietta entrambi i set
       per vedere se si sovrappongono nello stesso spazio a bassa dimensione.
    d) TEST DEL CLASSIFICATORE: Tenta di addestrare un classificatore per
       distinguere i due set. Un'accuratezza vicina al 50% indica alta somiglianza.

Come interpretare i risultati:
-   **Statistiche diverse**: Potrebbe esserci un problema di pre-processing (es. normalizzazione diversa).
-   **KDE non sovrapposti**: Le features individuali hanno distribuzioni diverse.
-   **Cluster separati in PCA**: C'è una differenza strutturale/lineare tra i due set.
-   **Accuratezza del classificatore alta**: Esiste una differenza sistematica (lineare o non)
    che rende i due set distinguibili. Questo è l'indicatore più forte di incompatibilità.
"""
import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Diagnostic script to compare two feature sets.')
    parser.add_argument('--features_large_path', type=str, required=True,
                        help='Path to the LARGE features file (e.g., millions, .npy or .h5).')
    parser.add_argument('--features_small_path', type=str, required=True,
                        help='Path to the SMALL features file (e.g., thousands, .h5 or .npy).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output plots.')
    parser.add_argument('--n_features_to_plot', type=int, default=9,
                        help='Number of random individual features to plot for distribution comparison.')
    return parser.parse_args()

def load_features(features_path):
    """Loads features from either an HDF5 or NumPy file."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading data from: {features_path}")
    if features_path.endswith(('.h5', '.hdf5')):
        with h5py.File(features_path, 'r') as h5f:
            if 'features' not in h5f: raise KeyError(f"'features' not found in: {features_path}")
            features = h5f['features'][:]
    elif features_path.endswith('.npy'):
        features = np.load(features_path, mmap_mode='r')
    else:
        raise ValueError(f"Unsupported file format: {features_path}. Use .h5, .hdf5, or .npy.")
    
    # Assicuriamoci che sia un array in memoria per le operazioni successive
    return np.array(features)

def analyze_summary_stats(features_a, name_a, features_b, name_b):
    """Prints a comparison of basic statistics."""
    print("\n--- 1. Analisi delle Statistiche di Base ---")
    
    stats_a = {
        "mean": np.mean(features_a),
        "std": np.std(features_a),
        "min": np.min(features_a),
        "max": np.max(features_a)
    }
    stats_b = {
        "mean": np.mean(features_b),
        "std": np.std(features_b),
        "min": np.min(features_b),
        "max": np.max(features_b)
    }

    print(f"{'Statistica':<10} | {'Set ' + name_a:<20} | {'Set ' + name_b:<20}")
    print("-" * 55)
    for key in stats_a:
        print(f"{key:<10} | {stats_a[key]:<20.4f} | {stats_b[key]:<20.4f}")
    
    if abs(stats_a['mean'] - stats_b['mean']) > 1e-2 or abs(stats_a['std'] - stats_b['std']) > 1e-2:
        print("\n[ATTENZIONE] Le statistiche di base (media/std) sono significativamente diverse.")
        print("Questo potrebbe indicare un diverso pre-processing (es. standardizzazione) e causare problemi in UMAP.")
    else:
        print("\n[OK] Le statistiche di base sembrano compatibili.")


def plot_feature_distributions(features_a, name_a, features_b, name_b, n_to_plot, output_path):
    """Plots the Kernel Density Estimate for a random subset of features."""
    print(f"\n--- 2. Visualizzazione di {n_to_plot} Distribuzioni di Features Casuali ---")
    
    # Scegli indici casuali delle features da plottare
    num_features = features_a.shape[1]
    feature_indices = np.random.choice(num_features, min(n_to_plot, num_features), replace=False)
    
    n_cols = 3
    n_rows = (len(feature_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(feature_indices):
        sns.kdeplot(features_a[:, idx], ax=axes[i], label=name_a, fill=True, alpha=0.5, clip=(-5, 5))
        sns.kdeplot(features_b[:, idx], ax=axes[i], label=name_b, fill=True, alpha=0.5, clip=(-5, 5))
        axes[i].set_title(f'Distribuzione Feature #{idx}')
        axes[i].legend()

    # Nascondi assi extra
    for i in range(len(feature_indices), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Confronto Distribuzioni di Features Casuali", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot salvato in: {output_path}")

def plot_pca_projection(features_a, name_a, features_b, name_b, output_path):
    """Trains PCA on set A, then transforms both A and B and plots them."""
    print("\n--- 3. Analisi di Proiezione Incrociata (PCA) ---")

    # 1. Standardizza i dati (è buona pratica prima della PCA)
    #    Addestra lo scaler SOLO sul set A e applicalo a entrambi
    scaler = StandardScaler()
    features_a_scaled = scaler.fit_transform(features_a)
    features_b_scaled = scaler.transform(features_b)

    # 2. Addestra la PCA SOLO sul set A
    pca = PCA(n_components=2)
    pca.fit(features_a_scaled)

    # 3. Trasforma entrambi i set usando la PCA addestrata su A
    features_a_pca = pca.transform(features_a_scaled)
    features_b_pca = pca.transform(features_b_scaled)
    
    # 4. Plotta
    plt.figure(figsize=(12, 10))
    plt.scatter(features_a_pca[:, 0], features_a_pca[:, 1], s=5, alpha=0.2, label=f'Set {name_a} (Training)')
    plt.scatter(features_b_pca[:, 0], features_b_pca[:, 1], s=10, alpha=0.7, label=f'Set {name_b} (Proiettato)')
    plt.title("Proiezione PCA: Addestrata su A, applicata a entrambi")
    plt.xlabel("Componente Principale 1")
    plt.ylabel("Componente Principale 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Se i punti blu e arancioni si sovrappongono, le loro strutture principali sono simili.")
    print("Se formano due cluster separati, le loro distribuzioni sono diverse.")
    print(f"Plot salvato in: {output_path}")

def run_classifier_test(features_a, features_b):
    """Trains a classifier to distinguish between the two sets."""
    print("\n--- 4. Test del Classificatore a 2 Campioni ---")
    
    # 1. Combina i dati e crea le etichette (0 per A, 1 per B)
    X = np.vstack([features_a, features_b])
    y = np.hstack([np.zeros(len(features_a)), np.ones(len(features_b))])
    
    # 2. Standardizza i dati combinati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Usa un classificatore semplice e veloce (Regressione Logistica)
    #    con cross-validation per una stima robusta dell'accuratezza.
    clf = LogisticRegression(solver='liblinear', random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    
    mean_accuracy = np.mean(scores)
    print(f"Accuratezza media (5-fold CV) del classificatore nel distinguere i due set: {mean_accuracy:.2%}")
    
    if mean_accuracy < 0.60:
        print("[OTTIMO] L'accuratezza è vicina al 50% (casuale). I set sono molto difficili da distinguere, quindi le loro distribuzioni sono probabilmente compatibili.")
    elif mean_accuracy < 0.85:
        print("[ATTENZIONE] Il classificatore riesce a distinguere i set con una certa abilità. Ci sono differenze, ma potrebbero essere sottili.")
    else:
        print("[PROBLEMA] L'accuratezza è alta. Il classificatore distingue facilmente i due set. Le loro distribuzioni sono quasi certamente INCOMPATIBILI.")


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Carica i dati
        features_large = load_features(args.features_large_path)
        features_small = load_features(args.features_small_path)

        print(f"\nCaricamento completato:")
        print(f"  - Set Grande ('A'): {features_large.shape}")
        print(f"  - Set Piccolo ('B'): {features_small.shape}")
        
        if features_large.shape[1] != features_small.shape[1]:
            print(f"[ERRORE] Le dimensioni delle features non corrispondono: {features_large.shape[1]} vs {features_small.shape[1]}")
            return

        # Sottocampiona il set grande per avere una dimensione comparabile a quello piccolo
        # Questo è cruciale per confronti visivi (KDE, PCA) e per il test del classificatore
        print(f"\nSottocampionamento del set grande a {len(features_small)} campioni per un confronto equo...")
        sample_indices = np.random.choice(features_large.shape[0], len(features_small), replace=False)
        features_large_sampled = features_large[sample_indices]

        # Esegui le analisi
        analyze_summary_stats(features_large_sampled, "A (Large, Sampled)", features_small, "B (Small)")
        
        plot_feature_distributions(
            features_large_sampled, "A (Large, Sampled)",
            features_small, "B (Small)",
            args.n_features_to_plot,
            os.path.join(args.output_dir, "feature_distribution_comparison.png")
        )
        
        plot_pca_projection(
            features_large_sampled, "A (Large, Sampled)",
            features_small, "B (Small)",
            os.path.join(args.output_dir, "pca_cross_projection.png")
        )

        run_classifier_test(features_large_sampled, features_small)

        print("\n--- Analisi Completata ---")

    except Exception as e:
        print(f"\n--- È avvenuto un errore ---")
        print(f"Tipo di errore: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()