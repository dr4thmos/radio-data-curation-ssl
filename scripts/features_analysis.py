import numpy as np
import matplotlib.pyplot as plt
import os
from cuml import UMAP, PCA, KMeans
import cudf
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pandas as pd

# Directory per salvare output
output_dir = "/path/to/output/"
os.makedirs(output_dir, exist_ok=True)

# Carica i dataset di features (modificare i percorsi)
feature_sets = {
    "network1": np.load("/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__cc6c34d6/features.npy"),
    "network2": np.load("/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__c1e974fa/features.npy"),
    "network3": np.load("/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__64effbed/features.npy")
}

# 1. Analisi dimensionale e statistiche di base
stats_report = {}
for name, features in feature_sets.items():
    # Statistiche di base
    stats = {
        "mean": np.mean(features, axis=0),
        "std": np.std(features, axis=0),
        "min": np.min(features, axis=0),
        "max": np.max(features, axis=0),
        "sparsity": np.mean(features == 0)
    }
    stats_report[name] = stats
    
    # Salva istogramma della distribuzione di valori
    plt.figure(figsize=(10, 6))
    plt.hist(features.flatten(), bins=100, alpha=0.7)
    plt.title(f"Distribuzione valori features - {name}")
    plt.savefig(f"{output_dir}/{name}_distribution.png")
    plt.close()

np.save(f"{output_dir}/stats_report.npy", stats_report)

# 2. Riduzione dimensionale con GPU
for name, features in feature_sets.items():
    # PCA
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    np.save(f"{output_dir}/{name}_pca.npy", features_pca)
    
    # Visualizza primi 2 componenti PCA
    plt.figure(figsize=(10, 8))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.1, s=1)
    plt.title(f"PCA - {name}")
    plt.savefig(f"{output_dir}/{name}_pca_plot.png")
    plt.close()
    
    # UMAP
    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    features_umap = umap.fit_transform(features)
    np.save(f"{output_dir}/{name}_umap.npy", features_umap)
    
    # Visualizza proiezione UMAP
    plt.figure(figsize=(10, 8))
    plt.scatter(features_umap[:, 0], features_umap[:, 1], alpha=0.1, s=1)
    plt.title(f"UMAP - {name}")
    plt.savefig(f"{output_dir}/{name}_umap_plot.png")
    plt.close()

# 3. Clustering
for name, features in feature_sets.items():
    # K-means con GPU
    n_clusters = 10  # Da regolare in base ai dati
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    np.save(f"{output_dir}/{name}_kmeans_labels.npy", labels)
    
    # Visualizza centroidi se possibile
    centroids = kmeans.cluster_centers_
    np.save(f"{output_dir}/{name}_kmeans_centroids.npy", centroids)

# 4. Image Retrieval con FAISS
# Esempio: costruisci indice per ricerca nearest neighbor
sample_size = features.shape[0] #min(3000000, )  # Limita campione per memoria
for name, features in feature_sets.items():
    #features_sample = features[:sample_size].astype('float32')
    features_gpu = features.astype('float32')
    d = features_gpu.shape[1]  # dimensionalità
    
    # Indice GPU FAISS
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(features_gpu)
    
    # Salva la matrice di distanza per i primi N elementi
    n_query = 1000
    k = 10  # Numero di vicini
    distances, indices = gpu_index.search(features_gpu[:n_query], k)
    np.save(f"{output_dir}/{name}_nn_distances.npy", distances)
    np.save(f"{output_dir}/{name}_nn_indices.npy", indices)

# 5. Confronto tra reti: similarità coseno tra features medie
mean_features = {name: np.mean(features, axis=0) for name, features in feature_sets.items()}
mean_features_matrix = np.array(list(mean_features.values()))
similarity_matrix = cosine_similarity(mean_features_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, xticklabels=list(feature_sets.keys()), 
            yticklabels=list(feature_sets.keys()))
plt.title("Similarità tra features medie delle diverse reti")
plt.savefig(f"{output_dir}/network_similarity.png")
plt.close()

# 6. Report finale
with open(f"{output_dir}/analysis_report.txt", "w") as f:
    f.write("Analisi Features Radioastronomiche\n")
    f.write("================================\n\n")
    
    for name in feature_sets.keys():
        f.write(f"Network: {name}\n")
        f.write(f"- Dimensione: {feature_sets[name].shape}\n")
        f.write(f"- Media: {np.mean(stats_report[name]['mean']):.4f}\n")
        f.write(f"- Deviazione std: {np.mean(stats_report[name]['std']):.4f}\n")
        f.write(f"- Sparsità: {stats_report[name]['sparsity']:.4f}\n\n")
    
    f.write("Visualizzazioni salvate:\n")
    f.write("- Distribuzioni, PCA e UMAP per ogni rete\n")
    f.write("- Matrice similarità tra reti\n")