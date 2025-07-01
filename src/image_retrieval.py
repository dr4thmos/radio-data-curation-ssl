import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def load_features(features_path):
    """Load pre-computed features from a numpy file."""
    return np.load(features_path)

def load_image_paths(info_json_path):
    """Load image paths from the info.json file."""
    with open(info_json_path, 'r') as f:
        info = json.load(f)
    return info['image_paths'] if 'image_paths' in info else info

def find_similar_images(query_feature, features, top_k=5):
    """Find the top-k similar images based on cosine similarity."""
    # Reshape query if needed
    if len(query_feature.shape) == 1:
        query_feature = query_feature.reshape(1, -1)
    
    # Calculate similarity
    similarities = cosine_similarity(query_feature, features)
    
    # Get indices of top-k similar images
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    
    return top_indices, similarities[0][top_indices]

def show_results(image_paths, query_idx, top_indices, similarities):
    """Display the query image and its top-k similar images."""
    plt.figure(figsize=(15, 3))
    
    # Display query image
    plt.subplot(1, len(top_indices) + 1, 1)
    query_img = Image.open(image_paths[query_idx])
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis('off')
    
    # Display similar images
    for i, (idx, sim) in enumerate(zip(top_indices, similarities)):
        plt.subplot(1, len(top_indices) + 1, i + 2)
        img = Image.open(image_paths[idx])
        plt.imshow(img)
        plt.title(f"Sim: {sim:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_retrieval_pdf(features_path, info_json_path, query_indices, output_pdf_path, top_k=5):
    """
    Create a PDF with multiple query images and their retrieval results.
    
    Args:
        features_path: Path to the features.npy file
        info_json_path: Path to the info.json file with image paths
        query_indices: List of indices of query images
        output_pdf_path: Path where to save the PDF
        top_k: Number of similar images to retrieve for each query
    """
    # Load features and image paths
    features = load_features(features_path)
    image_paths = load_image_paths(info_json_path)
    
    # Create PDF
    with PdfPages(output_pdf_path) as pdf:
        for query_idx in query_indices:
            # Get query feature
            query_feature = features[query_idx]
            
            # Find similar images
            top_indices, similarities = find_similar_images(query_feature, features, top_k)
            
            # Create a figure for this query
            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, top_k + 1, width_ratios=[1] + [1] * top_k)
            
            # Display query image
            ax = plt.subplot(gs[0])
            query_img = Image.open(image_paths[query_idx])
            ax.imshow(query_img)
            ax.set_title("Query")
            ax.axis('off')
            
            # Display similar images
            for i, (idx, sim) in enumerate(zip(top_indices, similarities)):
                ax = plt.subplot(gs[i + 1])
                img = Image.open(image_paths[idx])
                ax.imshow(img)
                ax.set_title(f"Sim: {sim:.3f}")
                ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
        # Add a summary page with info
        fig = plt.figure(figsize=(8, 4))
        plt.axis('off')
        summary_text = f"Image Retrieval Summary\n\n" \
                      f"Number of query images: {len(query_indices)}\n" \
                      f"Similar images per query: {top_k}\n" \
                      f"Features file: {os.path.basename(features_path)}\n" \
                      f"Total dataset size: {len(features)} images"
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"PDF saved to {output_pdf_path}")

def image_retrieval_system(features_path, info_json_path, query_idx=None, query_feature=None, top_k=5):
    """
    Main function for image retrieval.
    
    Args:
        features_path: Path to the features.npy file
        info_json_path: Path to the info.json file with image paths
        query_idx: Index of the query image (if using an existing image as query)
        query_feature: Feature vector of the query (if using a custom query)
        top_k: Number of similar images to retrieve
    """
    # Load features and image paths
    features = load_features(features_path)
    image_paths = load_image_paths(info_json_path)
    
    # Get query feature
    if query_feature is None and query_idx is not None:
        query_feature = features[query_idx]
    elif query_feature is None and query_idx is None:
        # Default to first image if no query specified
        query_idx = 0
        query_feature = features[query_idx]
    
    # Find similar images
    top_indices, similarities = find_similar_images(query_feature, features, top_k)
    
    # Show results
    show_results(image_paths, query_idx, top_indices, similarities)
    
    return top_indices, similarities

if __name__ == "__main__":
    query_indices = [100, 100000, 200000, 3000000, 567834]  # Indices of query images
    output_pdf_path = "image_retrieval_results.pdf"
    create_retrieval_pdf(features_path, info_json_path, query_indices, output_pdf_path, top_k=5)

# Example usage:
# 1. For single image retrieval:
# features_path = "outputs/features/features.npy"
# info_json_path = "path/to/info.json"  # This path will be retrieved via MLflow ID
# top_indices, similarities = image_retrieval_system(features_path, info_json_path, query_idx=0, top_k=5)

# 2. For creating a PDF with multiple query images:
# query_indices = [0, 10, 20, 30, 40]  # Indices of query images
# output_pdf_path = "image_retrieval_results.pdf"
# create_retrieval_pdf(features_path, info_json_path, query_indices, output_pdf_path, top_k=5)