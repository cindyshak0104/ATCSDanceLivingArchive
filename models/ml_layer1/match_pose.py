"""
STEP3: ML Models - Layer 1 (Body Similarity)
 3.1.1: Pose Matching (THIS FILE)
 3.1.2: Feature extraction from pose landmarks
 3.1.3: Similarity scoring between poses

PURPOSE: Core pose matching engine using ML-based embeddings.
Compares query poses against an archive of dance poses using learned embeddings.
Implements cosine similarity and L2 distance metrics for ranking matches.
Used by STEP4 (backend.py) for real-time dance pose matching.
"""

"""
Match a new user pose against archive embeddings.

This script loads pre-computed archive embeddings and finds similar poses
for a given user pose using the trained encoder model.

Usage:
    from match_pose import PoseMatcher
    
    matcher = PoseMatcher()
    matches = matcher.match(user_pose, top_k=10)
    
    for match in matches:
        print(f"Match: {match['archive_index']}, "
              f"Similarity: {match['similarity']:.4f}, "
              f"Distance: {match['distance']:.4f}")
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from pose_model import PoseEncoder


class PoseMatcher:
    """Match user poses against archive of poses."""
    
    def __init__(
        self,
        archive_csv_path=None,
        model_path=None,
        embeddings_path=None,
        metadata_path=None,
        device=None,
        metric="cosine",
        min_visibility=0.3,
    ):
        """
        Initialize the pose matcher.
        
        Args:
            archive_csv_path: Path to archive CSV (for metadata)
            model_path: Path to trained model weights
            embeddings_path: Path to pre-computed archive embeddings
            metadata_path: Path to archive metadata CSV
            device: torch device ('cuda' or 'cpu'). If None, auto-detect.
        """
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        
        # Set default paths
        if model_path is None:
            model_path = script_dir / "saved_models" / "pose_encoder.pt"
        else:
            model_path = Path(model_path)
        
        if embeddings_path is None:
            embeddings_path = script_dir / "saved_models" / "archive_embeddings.npy"
        else:
            embeddings_path = Path(embeddings_path)
        
        if metadata_path is None:
            metadata_path = script_dir / "saved_models" / "archive_metadata.csv"
        else:
            metadata_path = Path(metadata_path)
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = PoseEncoder(input_dim=66, embedding_dim=32)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"✓ Loaded model from {model_path}")
        
        # Load embeddings
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Archive embeddings not found at {embeddings_path}. "
                f"Run build_archive_embeddings.py first."
            )
        
        self.archive_embeddings = np.load(embeddings_path)
        print(f"✓ Loaded {len(self.archive_embeddings)} archive embeddings")
        
        # Load metadata if available
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = None
        
        # Load original CSV if available (for additional metadata)
        if archive_csv_path is None:
            # Try to find archive CSV
            archive_options = [
                project_root / "archive" / "pose_data.csv",
                project_root / "archive_poses.csv",
            ]
            for opt in archive_options:
                if opt.exists():
                    archive_csv_path = opt
                    break
        
        if archive_csv_path and Path(archive_csv_path).exists():
            self.archive_csv = pd.read_csv(archive_csv_path)
        else:
            self.archive_csv = None
        
        print(f"Using device: {self.device}")
        self.df = None
        self.metric = metric
        self.min_visibility = min_visibility

    def normalize_pose(self, pose):
        """Normalize pose to be translation and scale invariant."""
        pose = np.array(pose, dtype=np.float32)
        
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
        
        return pose.flatten()
    
    def get_embedding(self, pose):
        """
        Get embedding for a single pose.
        
        Args:
            pose: numpy array of shape (66,) or (33, 2), a list of MediaPipe landmarks,
                or a MediaPipe pose object.
        
        Returns:
            embedding: numpy array of shape (32,)
        """
        pose = self._pose_to_array(pose)

        # Normalize
        pose = self.normalize_pose(pose)
        
        # Convert to tensor
        pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
        pose_tensor = pose_tensor.to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(pose_tensor)
        
        return embedding.squeeze(0).cpu().numpy()

    def match(self, user_pose, top_k=10, metric="cosine"):
        """
        Find most similar poses in archive.
        
        Args:
            user_pose: numpy array of shape (66,) or (33, 2)
            top_k: number of top matches to return
            metric: 'cosine' (default) or 'l2' (L2 distance)
                   - cosine: uses cosine similarity (dot product of normalized embeddings)
                   - l2: uses L2 distance (Euclidean distance)
        
        Returns:
            List of dicts with keys:
                - archive_index: index in archive
                - similarity: similarity score (cosine: 0-1, l2: 0-∞)
                - distance: L2 distance (always computed)
                - metadata: additional metadata if available
        """
        # Get user embedding
        user_embedding = self.get_embedding(user_pose)
        
        if metric == "cosine":
            # Cosine similarity: dot product of normalized embeddings
            # (embeddings already L2-normalized from model)
            similarities = np.dot(self.archive_embeddings, user_embedding)
            
            # Sort by similarity (descending)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                # Also compute L2 distance for reference
                distance = np.linalg.norm(self.archive_embeddings[idx] - user_embedding)
                
                result = {
                    "archive_index": int(idx),
                    "similarity": float(similarities[idx]),  # Cosine similarity
                    "distance": float(distance),              # L2 distance
                }
                
                # Add metadata if available
                if self.metadata is not None and idx < len(self.metadata):
                    metadata_row = self.metadata.iloc[idx]
                    result["metadata"] = metadata_row.to_dict()
                
                results.append(result)
        
        elif metric == "l2":
            # L2 distance: Euclidean distance
            distances = np.linalg.norm(
                self.archive_embeddings - user_embedding,
                axis=1
            )
            
            # Sort by distance (ascending)
            top_indices = np.argsort(distances)[:top_k]
            
            results = []
            for idx in top_indices:
                # Also compute cosine similarity for reference
                cosine_sim = np.dot(self.archive_embeddings[idx], user_embedding)
                
                result = {
                    "archive_index": int(idx),
                    "distance": float(distances[idx]),         # L2 distance
                    "similarity": float(cosine_sim),           # Cosine similarity
                }
                
                # Add metadata if available
                if self.metadata is not None and idx < len(self.metadata):
                    metadata_row = self.metadata.iloc[idx]
                    result["metadata"] = metadata_row.to_dict()
                
                results.append(result)
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'l2'.")
        
        return results

    def _pose_to_array(self, pose):
        if hasattr(pose, "landmark"):
            landmark_list = pose.landmark
        elif isinstance(pose, (list, tuple)) and pose and hasattr(pose[0], "x"):
            landmark_list = pose
        else:
            landmark_list = None

        if landmark_list is not None:
            coords = np.zeros((len(landmark_list[:33]), 2), dtype=np.float32)
            for idx, landmark in enumerate(landmark_list[:33]):
                coords[idx, 0] = float(landmark.x)
                coords[idx, 1] = float(landmark.y)
            return coords

        pose_array = np.asarray(pose, dtype=np.float32)

        if pose_array.ndim == 1 and pose_array.shape[0] == 66:
            return pose_array.reshape(33, 2)

        if pose_array.ndim == 2 and pose_array.shape[1] == 2 and pose_array.shape[0] == 33:
            return pose_array

        raise ValueError(
            f"Expected a MediaPipe pose, a list of landmarks, a 66-value array, or a (33, 2) array; got shape {pose_array.shape}"
        )

    def _row_to_landmark_array(self, row):
        coords = np.zeros((33, 2), dtype=np.float32)
        for idx in range(33):
            coords[idx, 0] = float(row[f"lm{idx}_x"])
            coords[idx, 1] = float(row[f"lm{idx}_y"])
        return coords

    def _embed_dataframe(self, df):
        embeddings = []
        for _, row in df.iterrows():
            embedding = self.get_embedding(self._row_to_landmark_array(row))
            embeddings.append(embedding)
        return np.stack(embeddings, axis=0)

    def fit_from_dataframe(self, df):
        if df is None:
            raise ValueError("fit_from_dataframe requires a pandas DataFrame")

        required_columns = [f"lm{i}_x" for i in range(33)] + [f"lm{i}_y" for i in range(33)]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required landmark columns: {missing_columns}")

        self.df = df.reset_index(drop=True).copy()
        self.archive_embeddings = self._embed_dataframe(self.df)
        return self

    def fit_from_csv(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return self.fit_from_dataframe(df)

    def match_top_k(self, pose_landmarks, k=12, metric=None):
        if self.df is None:
            raise ValueError("The matcher has not been fitted. Call fit_from_dataframe or fit_from_csv first.")

        if metric is None:
            metric = self.metric

        query_pose = self._pose_to_array(pose_landmarks)
        matches = self.match(query_pose, top_k=k, metric=metric)

        results = []
        for match in matches:
            row_index = int(match["archive_index"])
            row = self.df.iloc[row_index]
            stick_figure_path = row.get("stick_figure_path") if "stick_figure_path" in row else None

            results.append(
                SimpleNamespace(
                    row_index=row_index,
                    pose_id=str(row.get("pose_id", f"archive_{row_index}")),
                    distance=float(match["distance"]),
                    similarity=float(match["similarity"]),
                    stick_figure_path=str(stick_figure_path) if pd.notna(stick_figure_path) else None,
                )
            )

        return results


def demo():
    """Demo: match a random pose against archive."""
    print("="*70)
    print("POSE MATCHING DEMO")
    print("="*70)
    
    # Initialize matcher
    print("\nInitializing matcher...")
    try:
        matcher = PoseMatcher()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease create archive embeddings first:")
        print("  python3 build_archive_embeddings.py")
        return
    
    # Create random test pose
    print("\nCreating random test pose...")
    test_pose = np.random.randn(66).astype(np.float32)
    
    # Find matches using cosine similarity
    print("\nFinding top-10 matches (cosine similarity)...")
    matches = matcher.match(test_pose, top_k=10, metric="cosine")
    
    print(f"\nTop-10 matches:")
    print(f"{'Rank':<6} {'Index':<8} {'Similarity':<12} {'Distance':<12}")
    print("-" * 40)
    
    for rank, match in enumerate(matches, 1):
        print(f"{rank:<6} {match['archive_index']:<8} "
              f"{match['similarity']:<12.4f} {match['distance']:<12.4f}")
    
    # Also show L2-based ranking
    print(f"\n\nTop-10 matches (L2 distance)...")
    matches_l2 = matcher.match(test_pose, top_k=10, metric="l2")
    
    print(f"\nTop-10 matches:")
    print(f"{'Rank':<6} {'Index':<8} {'Distance':<12} {'Similarity':<12}")
    print("-" * 40)
    
    for rank, match in enumerate(matches_l2, 1):
        print(f"{rank:<6} {match['archive_index']:<8} "
              f"{match['distance']:<12.4f} {match['similarity']:<12.4f}")
    
    print(f"\n{'='*70}")
    print("✓ Demo complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    demo()
