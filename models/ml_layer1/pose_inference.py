"""
STEP3: ML Models - Layer 1 (Body Similarity)
 3.1.1: Pose Matching
 3.1.2: Pose Inference / Feature Extraction (THIS FILE)
 3.1.3: Similarity scoring

PURPOSE: Converts raw pose landmarks into learned embeddings.
Loads the trained pose encoder model and generates fixed-size feature vectors.
These embeddings enable efficient similarity comparison in pose_matching.py
"""

"""
Inference utility for the pose encoder model.
Use this to find similar poses in your archive or embed new poses.
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pose_model import PoseEncoder


class PoseEmbedder:
    """Wrapper for using the pose encoder model for inference."""
    
    def __init__(self, model_path, device=None):
        """
        Load the trained pose encoder model.
        
        Args:
            model_path: Path to the saved model state_dict
            device: torch device ('cuda' or 'cpu'). If None, auto-detect.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = PoseEncoder(input_dim=66, embedding_dim=32)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {model_path} on {device}")
    
    def normalize_pose(self, pose):
        """
        Normalize a pose (66-dim vector of 33 landmarks with x,y coords).
        
        Args:
            pose: numpy array of shape (66,) or (33, 2)
        
        Returns:
            Normalized pose as numpy array of shape (66,)
        """
        if pose.shape == (66,):
            pose = pose.reshape(-1, 2)
        elif pose.shape != (33, 2):
            raise ValueError(f"Expected pose shape (66,) or (33, 2), got {pose.shape}")
        
        # Center by centroid
        center = np.mean(pose, axis=0)
        pose = pose - center
        
        # Scale by norm (size invariant)
        scale = np.linalg.norm(pose)
        if scale > 0:
            pose = pose / scale
        
        return pose.flatten().astype(np.float32)
    
    def embed_pose(self, pose):
        """
        Convert a pose to its embedding vector.
        
        Args:
            pose: numpy array of shape (66,) - raw keypoint coordinates
        
        Returns:
            Embedding vector as numpy array of shape (32,)
        """
        # Normalize the pose
        normalized = self.normalize_pose(pose)
        
        # Convert to tensor
        pose_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        pose_tensor = pose_tensor.to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(pose_tensor)
        
        return embedding.cpu().numpy()[0]  # Return as (32,) numpy array
    
    def embed_poses(self, poses):
        """
        Convert multiple poses to their embedding vectors.
        
        Args:
            poses: numpy array of shape (N, 66) - N poses
        
        Returns:
            Embeddings as numpy array of shape (N, 32)
        """
        embeddings = []
        
        for pose in poses:
            embedding = self.embed_pose(pose)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def find_similar_poses(self, query_pose, archive_embeddings, top_k=10):
        """
        Find the most similar poses in an archive.
        
        Args:
            query_pose: numpy array of shape (66,) - the query pose
            archive_embeddings: numpy array of shape (N, 32) - embeddings of archive poses
            top_k: number of top similar poses to return
        
        Returns:
            List of (index, distance) tuples, sorted by distance (ascending)
        """
        # Embed the query pose
        query_embedding = self.embed_pose(query_pose)
        
        # Compute L2 distances to all archive embeddings
        distances = np.linalg.norm(archive_embeddings - query_embedding, axis=1)
        
        # Get top-k most similar
        top_k_indices = np.argsort(distances)[:top_k]
        
        # Return as list of (index, distance)
        results = [(idx, distances[idx]) for idx in top_k_indices]
        return results


def example_usage():
    """Example of how to use the pose embedder."""
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / "saved_models" / "pose_encoder.pt"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    # Initialize embedder
    embedder = PoseEmbedder(str(model_path))
    
    # Example 1: Embed a single pose
    print("\nExample 1: Embed a single pose")
    print("-" * 60)
    random_pose = np.random.randn(66).astype(np.float32)
    embedding = embedder.embed_pose(random_pose)
    print(f"Pose shape: {random_pose.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f} (should be ~1.0)")
    
    # Example 2: Embed multiple poses
    print("\nExample 2: Embed multiple poses")
    print("-" * 60)
    random_poses = np.random.randn(10, 66).astype(np.float32)
    embeddings = embedder.embed_poses(random_poses)
    print(f"Poses shape: {random_poses.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Example 3: Find similar poses
    print("\nExample 3: Find similar poses")
    print("-" * 60)
    query_pose = np.random.randn(66).astype(np.float32)
    archive_embeddings = np.random.randn(100, 32).astype(np.float32)
    # Normalize archive embeddings to unit norm
    archive_embeddings = archive_embeddings / (np.linalg.norm(archive_embeddings, axis=1, keepdims=True) + 1e-6)
    
    results = embedder.find_similar_poses(query_pose, archive_embeddings, top_k=5)
    print(f"Query pose shape: {query_pose.shape}")
    print(f"Archive size: {len(archive_embeddings)}")
    print(f"\nTop-5 similar poses:")
    for rank, (idx, distance) in enumerate(results, 1):
        print(f"  {rank}. Pose #{idx}: distance = {distance:.4f}")


if __name__ == "__main__":
    example_usage()
