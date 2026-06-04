"""
STEP3: ML Models - Layer 2 (Emotional Similarity)
 3.2.1: Emotion Model Architecture
 3.2.2: Feature Extraction (THIS FILE)
 3.2.3: Emotion prediction and reranking

PURPOSE: Extracts features from pose sequences for emotion/style classification.
Processes pose CSV files and generates feature vectors for training/inference.
Handles sequence padding and normalization for temporal pose data.
"""

"""
Extract aggregated emotion features from pose landmark data.
This module processes raw pose data and aggregates it by movement frame
to create input features and labels for the emotion prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


def load_and_process_csv(csv_path: str) -> pd.DataFrame:
    """Load a single CSV file and return the dataframe."""
    return pd.read_csv(csv_path)


def aggregate_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pose landmarks by frame.
    
    Groups all landmarks within a frame and computes statistics:
    - Position stats (mean, std of x, y, z)
    - Visibility stats (mean visibility)
    - Motion features (derived from position changes)
    
    Args:
        df: Raw pose dataframe with landmarks
    
    Returns:
        Aggregated dataframe with one row per frame
    """
    
    agg_dict = {
        'x': ['mean', 'std', 'min', 'max'],
        'y': ['mean', 'std', 'min', 'max'],
        'z': ['mean', 'std', 'min', 'max'],
        'visibility': ['mean'],
        'presence': ['mean'],
    }
    
    grouped = df.groupby('frame_index').agg(agg_dict)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Keep categorical columns from first frame
    categorical_cols = df.groupby('frame_index').agg({
        'emotion_primary': 'first',
        'emotion_intensity': 'first',
        'arousal': 'first',
        'tension': 'first',
        'weight': 'first',
        'flow': 'first',
        'memory_tag': 'first',
        'video_name': 'first',
        'movement': 'first'
    })
    
    result = grouped.join(categorical_cols)
    return result.reset_index()


def compute_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute motion-derived features from aggregated frame data.
    
    Features:
    - velocity: frame-to-frame position change
    - acceleration: velocity change
    - body_expansion: max - min values
    - vertical_level: mean y position
    - spatial_distribution: std of positions
    """
    
    df = df.copy()
    
    # Compute velocity (frame-to-frame change)
    df['x_mean_velocity'] = df['x_mean'].diff().abs()
    df['y_mean_velocity'] = df['y_mean'].diff().abs()
    df['z_mean_velocity'] = df['z_mean'].diff().abs()
    df['avg_velocity'] = df[['x_mean_velocity', 'y_mean_velocity', 'z_mean_velocity']].mean(axis=1)
    
    # Compute acceleration
    df['avg_acceleration'] = df['avg_velocity'].diff().abs()
    
    # Body expansion features
    df['body_expansion_x'] = df['x_max'] - df['x_min']
    df['body_expansion_y'] = df['y_max'] - df['y_min']
    df['body_expansion_z'] = df['z_max'] - df['z_min']
    df['total_expansion'] = df[['body_expansion_x', 'body_expansion_y', 'body_expansion_z']].mean(axis=1)
    
    # Vertical level (y position)
    df['vertical_level'] = df['y_mean']
    
    # Spatial spread
    df['spatial_spread'] = df[['x_std', 'y_std', 'z_std']].mean(axis=1)
    
    # Smoothness (inverse of acceleration)
    df['smoothness'] = 1.0 / (df['avg_acceleration'] + 0.01)
    
    # Fill NaN values created by diff()
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def create_feature_label_pairs(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Create feature-label pairs for training.
    
    Regression targets: emotion_intensity, arousal, tension, weight, flow
    Classification targets: emotion_primary, memory_tag
    
    Args:
        df: Processed dataframe with motion features
    
    Returns:
        (features_array, labels_dict)
    """
    
    # Define pose-based input features
    feature_cols = [
        'x_mean', 'x_std', 'x_min', 'x_max',
        'y_mean', 'y_std', 'y_min', 'y_max',
        'z_mean', 'z_std', 'z_min', 'z_max',
        'visibility_mean', 'presence_mean',
        'x_mean_velocity', 'y_mean_velocity', 'z_mean_velocity',
        'avg_velocity', 'avg_acceleration',
        'body_expansion_x', 'body_expansion_y', 'body_expansion_z', 'total_expansion',
        'vertical_level', 'spatial_spread', 'smoothness'
    ]
    
    # Regression label targets
    regression_cols = [
        'emotion_intensity',
        'arousal',
        'tension',
        'weight',
        'flow'
    ]
    
    # Classification label targets
    classification_cols = [
        'emotion_primary',
        'memory_tag'
    ]
    
    # Extract features (handle missing columns gracefully)
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_feature_cols].values.astype(np.float32)
    
    # Create labels dict
    labels = {}
    
    # Regression labels
    for col in regression_cols:
        if col in df.columns:
            labels[col] = df[col].values.astype(np.float32)
    
    # Classification labels (need encoding)
    for col in classification_cols:
        if col in df.columns:
            unique_vals = df[col].unique()
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            labels[col] = df[col].map(val_to_idx).values.astype(np.int64)
            labels[f'{col}_classes'] = len(unique_vals)
            labels[f'{col}_mapping'] = val_to_idx
    
    labels['feature_names'] = available_feature_cols
    labels['regression_targets'] = regression_cols
    labels['classification_targets'] = classification_cols
    
    return X, labels


def process_all_csvs(csv_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Process all CSV files in the directory and combine them.
    
    Args:
        csv_dir: Directory containing pose CSV files
    
    Returns:
        (combined_features, combined_labels)
    """
    
    csv_path = Path(csv_dir)
    csv_files = list(csv_path.glob('*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    
    all_X = []
    all_labels_dict = {}
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        # Load and process
        df = load_and_process_csv(str(csv_file))
        df = aggregate_by_frame(df)
        df = compute_motion_features(df)
        
        # Extract features and labels
        X, labels = create_feature_label_pairs(df)
        all_X.append(X)
        
        # Merge label information
        if not all_labels_dict:
            all_labels_dict = labels.copy()
        else:
            # Concatenate regression and classification arrays
            for key in labels:
                if isinstance(labels[key], np.ndarray):
                    if key in all_labels_dict and isinstance(all_labels_dict[key], np.ndarray):
                        all_labels_dict[key] = np.concatenate([all_labels_dict[key], labels[key]])
                    else:
                        all_labels_dict[key] = labels[key]
    
    combined_X = np.concatenate(all_X, axis=0)
    
    print(f"Combined dataset shape: {combined_X.shape}")
    print(f"Regression targets: {all_labels_dict.get('regression_targets', [])}")
    print(f"Classification targets: {all_labels_dict.get('classification_targets', [])}")
    
    return combined_X, all_labels_dict


if __name__ == "__main__":
    # Example usage
    csv_dir = "../pose_results_L2"
    X, labels = process_all_csvs(csv_dir)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features used: {labels['feature_names']}")
    
    # Show sample regression labels
    if 'emotion_intensity' in labels:
        print(f"Emotion intensity range: {labels['emotion_intensity'].min():.2f} to {labels['emotion_intensity'].max():.2f}")
    
    if 'arousal' in labels:
        print(f"Arousal range: {labels['arousal'].min():.2f} to {labels['arousal'].max():.2f}")
    
    # Show sample classification labels
    for col in labels.get('classification_targets', []):
        if f'{col}_classes' in labels:
            n_classes = labels[f'{col}_classes']
            print(f"{col}: {n_classes} classes")
