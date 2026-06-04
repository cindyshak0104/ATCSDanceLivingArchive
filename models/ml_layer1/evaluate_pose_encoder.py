"""
STEP3: ML Models - Layer 1 (Body Similarity)
UTILITY: Evaluate Pose Encoder (THIS FILE)

PURPOSE: Evaluation script for the pose encoder model.
Tests model performance on validation/test data.
Generates metrics and visualizations for model assessment.
"""

"""
Evaluation script for pose encoder model.
Tests the model's ability to find similar poses and measure embedding quality.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pose_model import PoseEncoder
from train_pose_encoder import PoseTripletDataset


def load_model(model_path, device, input_dim=66, embedding_dim=32):
    """Load trained model."""
    model = PoseEncoder(input_dim=input_dim, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_embeddings(model, dataloader, device):
    """Generate embeddings for all poses in dataloader."""
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, (anchor, _, _) in enumerate(dataloader):
            anchor = anchor.to(device)
            embedding = model(anchor)
            embeddings.append(embedding.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def compute_distances(embeddings):
    """Compute pairwise L2 distances between embeddings."""
    # embeddings shape: (N, embedding_dim)
    # Distance matrix: (N, N)
    n = embeddings.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        # Compute distances from embedding i to all others
        diffs = embeddings - embeddings[i]
        distances[i] = np.linalg.norm(diffs, axis=1)
    
    return distances


def evaluate_retrieval(embeddings, num_neighbors=10):
    """
    Evaluate pose retrieval quality.
    For each pose, check if augmented versions (same pose index) rank high.
    """
    n = len(embeddings)
    distances = compute_distances(embeddings)
    
    # Since we generated data where each pose has augmented versions,
    # we approximate by checking if nearest neighbors are nearby in dataset
    # (consecutive frames of same movement tend to be similar)
    
    # Compute recall@k for various k values
    recalls = {1: 0, 5: 0, 10: 0}
    
    for i in range(n):
        nearest_indices = np.argsort(distances[i])[1:11]  # Skip self
        
        # Check if nearby frames are in top-k
        for k in [1, 5, 10]:
            top_k = nearest_indices[:k]
            # Count how many neighbors are within reasonable distance (consecutive frames)
            nearby = sum(1 for j in top_k if abs(j - i) <= 5)
            recalls[k] += nearby / min(k, 5)  # Normalize
    
    # Average and print
    for k in [1, 5, 10]:
        avg_recall = recalls[k] / n
        print(f"  Recall@{k}: {avg_recall:.3f}")


def compute_embedding_stats(embeddings):
    """Compute statistics about embeddings."""
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    min_norm = np.min(np.linalg.norm(embeddings, axis=1))
    max_norm = np.max(np.linalg.norm(embeddings, axis=1))
    
    # All should be normalized to 1.0
    print(f"  Embedding norm - Mean: {mean_norm:.4f}, Min: {min_norm:.4f}, Max: {max_norm:.4f}")
    
    # Check embedding diversity
    centered = embeddings - embeddings.mean(axis=0)
    cov_matrix = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    effective_rank = np.sum(eigenvalues > 0.01 * np.max(eigenvalues))
    print(f"  Effective rank of embeddings: {effective_rank}/{embeddings.shape[1]}")


def test_single_query(model, device):
    """Test retrieving similar poses for a single query."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    csv_dir = project_root / "pose_results"
    
    dataset = PoseTripletDataset(str(csv_dir))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get embeddings
    print("\nGenerating embeddings for all poses...")
    embeddings = get_embeddings(model, dataloader, device)
    
    # Test retrieval with a few random poses
    print("\n" + "="*60)
    print("Testing retrieval quality:")
    print("="*60)
    
    num_tests = 5
    for test_idx in range(num_tests):
        query_idx = np.random.randint(0, len(embeddings))
        query_embedding = embeddings[query_idx]
        
        # Compute distances to all other embeddings
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        
        # Get top-10 most similar
        top_k = 10
        most_similar = np.argsort(distances)[:top_k]
        
        print(f"\nQuery pose {query_idx}:")
        print(f"  Top-{top_k} similar poses (index: distance):")
        for rank, idx in enumerate(most_similar):
            print(f"    {rank+1}. Pose {idx}: {distances[idx]:.4f}")


def main():
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    csv_dir = project_root / "pose_results"
    model_path = script_dir / "saved_models" / "pose_encoder.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python3 train_pose_encoder.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    print(f"Model loaded from {model_path}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = PoseTripletDataset(str(csv_dir))
    # Use a sample for evaluation (full dataset is large)
    sample_size = min(5000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    sample_dataset = torch.utils.data.Subset(dataset, sample_indices)
    dataloader = DataLoader(sample_dataset, batch_size=64, shuffle=False)
    print(f"Evaluating on {sample_size} poses\n")
    
    # Get embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(model, dataloader, device)
    
    # Compute statistics
    print("\nEmbedding Statistics:")
    print("-" * 60)
    compute_embedding_stats(embeddings)
    
    # Evaluate retrieval
    print("\nRetrieval Quality:")
    print("-" * 60)
    evaluate_retrieval(embeddings)
    
    # Test single query
    test_single_query(model, device)
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)
    print(f"\nModel is ready for pose similarity matching.")
    print(f"Use it with: model = load_model('{model_path}', device)")


if __name__ == "__main__":
    main()
