"""
STEP3: ML Models - Layer 1 (Body Similarity)
UTILITY: Build Archive Embeddings (THIS FILE)

PURPOSE: Preprocesses the pose archive database.
Extracts pose embeddings for all dances in the archive using the trained pose encoder.
Stores embeddings for fast retrieval during real-time matching in backend.py
"""

"""
Build archive embeddings from pose data.

This script loads archive pose data and generates embeddings using the trained model.
The embeddings are saved for fast similarity matching.

Usage:
    python3 build_archive_embeddings.py
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pose_model import PoseEncoder


def normalize_pose(pose):
    """Normalize pose to be translation and scale invariant."""
    pose = pose.reshape(-1, 2)

    # Center pose around centroid
    center = np.mean(pose, axis=0)
    pose = pose - center

    # Scale pose so body size differences matter less
    scale = np.linalg.norm(pose)
    if scale > 0:
        pose = pose / scale

    return pose.flatten()


def load_archive_poses(csv_path=None):
    """
    Load archive poses from CSV file or directory.
    
    Supports two formats:
    1. Single CSV with one pose per row (wide format): columns like "landmark_0_x", "landmark_0_y", etc.
    2. Multiple CSVs with long format (landmark per row): group by frame/sample
    
    Args:
        csv_path: Path to CSV file or directory of CSVs. If None, searches for "archive" dir.
    
    Returns:
        poses: numpy array of shape (N, 66) - N poses with 33 landmarks × 2 coords
        metadata: pandas DataFrame with source information for each pose
    """
    # Determine input path
    if csv_path is None:
        # Look for archive directory
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        
        # Try common archive paths
        archive_options = [
            project_root / "archive" / "pose_data.csv",
            project_root / "archive_poses.csv",
            project_root / "pose_archive.csv",
        ]
        
        csv_path = None
        for opt in archive_options:
            if opt.exists():
                csv_path = opt
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                "Could not find archive CSV. "
                "Please provide csv_path parameter or create archive/pose_data.csv"
            )
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Archive not found at {csv_path}")
    
    print(f"Loading archive from {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if it's wide format (columns like "x_0", "y_0", etc.)
    coord_cols = [col for col in df.columns if ("_x" in col or "_y" in col)]
    
    if coord_cols:
        # Wide format - one pose per row
        print("Detected wide format (one pose per row)")
        poses = df[coord_cols].values.astype(np.float32)
        metadata = df.drop(columns=coord_cols)
        
        # Verify dimensions
        if poses.shape[1] != 66:
            print(f"Warning: Expected 66 dimensions, got {poses.shape[1]}")
        
        return poses, metadata
    
    # Check for long format (x, y, z columns with landmark_id)
    elif "x" in df.columns and "y" in df.columns and "landmark_id" in df.columns:
        # Long format - one landmark per row
        print("Detected long format (one landmark per row)")
        
        # Group by sample (assuming first non-coordinate column is sample identifier)
        # Look for common identifiers
        sample_cols = []
        for col in ["frame_index", "sample_id", "pose_id", "index"]:
            if col in df.columns:
                sample_cols.append(col)
        
        if not sample_cols:
            # If no explicit sample column, use row groups
            print("Warning: No sample identifier found, using row grouping")
            sample_cols = None
        
        poses_list = []
        metadata_list = []
        
        if sample_cols:
            for sample_id, group in df.groupby(sample_cols):
                landmarks = group.sort_values("landmark_id")[["x", "y"]].values
                
                if len(landmarks) == 33:
                    pose = landmarks.flatten().astype(np.float32)
                    poses_list.append(pose)
                    metadata_list.append({**dict(zip(sample_cols, [sample_id] if not isinstance(sample_id, tuple) else sample_id))})
        else:
            # Group every 33 rows as a pose
            for i in range(0, len(df), 33):
                group = df.iloc[i:i+33]
                if len(group) == 33:
                    landmarks = group.sort_values("landmark_id")[["x", "y"]].values
                    pose = landmarks.flatten().astype(np.float32)
                    poses_list.append(pose)
                    metadata_list.append({"sample_index": i // 33})
        
        poses = np.array(poses_list, dtype=np.float32)
        metadata = pd.DataFrame(metadata_list)
        
        print(f"Extracted {len(poses)} poses with 33 landmarks each")
        return poses, metadata
    
    else:
        raise ValueError(
            f"Could not identify pose format. "
            f"Expected either '_x/_y' columns or 'x/y/landmark_id' columns. "
            f"Found columns: {df.columns.tolist()}"
        )


def build_embeddings(csv_path=None, model_path=None, output_path=None):
    """
    Build and save embeddings for archive poses.
    
    Args:
        csv_path: Path to archive CSV
        model_path: Path to trained model
        output_path: Path to save embeddings
    """
    # Set default paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    
    if model_path is None:
        model_path = script_dir / "saved_models" / "pose_encoder.pt"
    else:
        model_path = Path(model_path)
    
    if output_path is None:
        output_path = script_dir / "saved_models" / "archive_embeddings.npy"
    else:
        output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"\n{'='*70}")
    print("Building Archive Embeddings")
    print(f"{'='*70}")
    
    # Load archive poses
    print(f"\n1. Loading archive poses...")
    poses, metadata = load_archive_poses(csv_path)
    print(f"   Loaded {len(poses)} poses, shape: {poses.shape}")
    
    # Load model
    print(f"\n2. Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    model = PoseEncoder(input_dim=66, embedding_dim=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"   Model loaded from {model_path}")
    
    # Generate embeddings
    print(f"\n3. Generating embeddings...")
    all_embeddings = []
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            # Normalize
            normalized = normalize_pose(pose)
            
            # Convert to tensor
            pose_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            pose_tensor = pose_tensor.to(device)
            
            # Get embedding
            embedding = model(pose_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()
            
            all_embeddings.append(embedding)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(poses)} poses...")
    
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    
    print(f"\n4. Saving embeddings...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    print(f"   ✓ Saved to {output_path}")
    print(f"   Shape: {all_embeddings.shape}")
    print(f"   Size: {all_embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata_path = output_path.with_name("archive_metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    print(f"   ✓ Metadata saved to {metadata_path}")
    
    print(f"\n{'='*70}")
    print("✓ Archive embeddings built successfully!")
    print(f"{'='*70}\n")
    
    return all_embeddings, metadata


if __name__ == "__main__":
    build_embeddings()
