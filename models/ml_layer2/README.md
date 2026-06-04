# Layer 2: Emotion and Memory Similarity Prediction Network

A multi-task supervised neural network that predicts emotion vectors and categorical labels from pose-derived motion features. Integrates with Layer 1 (body similarity) to perform combined pose + emotion matching.

## Overview

### What is Layer 2?

Layer 2 is a **multi-task learning system** that performs:
1. **Regression**: Predict continuous emotion values
   - `emotion_intensity` (0-1): How intense is the emotion
   - `arousal` (0-1): Energy/activation level
   - `tension` (0-1): Physical tension
   - `weight` (0-1): Movement quality (light vs heavy)
   - `flow` (0-1): Smooth continuity

2. **Classification**: Predict categorical emotion/memory labels
   - `emotion_primary`: Emotion category (e.g., "happy", "sad", "angry")
   - `memory_tag`: Memory association (e.g., "chill self-expression")

### How does it integrate with Layer 1?

```
Layer 1 (Body Similarity):
  Input: Pose landmarks
  Output: Top 5 physically similar movements
  
  
  
Layer 2 (Emotion Similarity):
  Input: Pose landmarks + Top 5 from Layer 1
  Output: Re-ranked matches by emotional closeness
  
  
  
Final Score = 0.6  body_similarity + 0.4  emotion_similarity
```

## Architecture

### Neural Network Design

```
Input (26 features) 
  
Encoder (Shared):
 256) + BatchNorm + ReLU
 128) + BatchNorm + ReLU
 64) + BatchNorm + ReLU
  
Task-Specific Heads:
 5) [emotion_intensity, arousal, tension, weight, flow]
 n_classes) [emotion_primary]
 n_classes) [memory_tag]
```

### Input Features (26 total)

Derived from aggregated pose landmarks:
- **Position statistics**: x_mean, x_std, x_min, x_max, y_mean, etc. (12 features)
- **Visibility/Presence**: visibility_mean, presence_mean (2 features)
- **Velocity**: x_mean_velocity, y_mean_velocity, z_mean_velocity, avg_velocity (4 features)
- **Acceleration**: avg_acceleration (1 feature)
- **Body expansion**: body_expansion_x, body_expansion_y, body_expansion_z, total_expansion (4 features)
- **Spatial features**: vertical_level, spatial_spread, smoothness (3 features)

## Files

### 1. `extract_emotion_features.py`
Extract and preprocess emotion features from raw pose CSV files.

**Key functions:**
- `load_and_process_csv(csv_path)`: Load pose data
- `aggregate_by_frame(df)`: Group landmarks by frame
- `compute_motion_features(df)`: Derive velocity, acceleration, expansion
- `create_feature_label_pairs(df)`: Extract features and labels
- `process_all_csvs(csv_dir)`: Combined pipeline for all CSVs

**Usage:**
```python
from extract_emotion_features import process_all_csvs

X, labels = process_all_csvs("../pose_results_L2")
# X: (num_samples, 25) feature matrix
# labels: dict with regression/classification targets
```

### 2. `emotion_model.py`
Neural network architecture definitions.

**Classes:**
- `EmotionNet`: Multi-task learning model (recommended)
- `EmotionNetSimple`: Single-task regression-only model

**Usage:**
```python
from emotion_model import EmotionNet
import torch

model = EmotionNet(
    input_dim=26,
    regression_output_dim=5,
    emotion_primary_classes=5,
    memory_tag_classes=10
)

x = torch.randn(8, 25)
outputs = model(x)
# outputs['regression']: (8, 5) emotion vectors
# outputs['emotion_primary_logits']: (8, 5) classification logits
# outputs['memory_tag_logits']: (8, 10) classification logits
```

### 3. `train_emotion_model.py`
Complete training pipeline with validation and early stopping.

**Key functions:**
- `EmotionDataset`: PyTorch Dataset class
- `create_dataloaders()`: Split data into train/val
- `train_epoch()`: Single training epoch
- `validate()`: Validation step
- `train_model()`: Full training loop

**Usage:**
```bash
cd ml_layer2
python train_emotion_model.py
```

**Outputs:**
- `../saved_models/emotion_model_best.pt`: Best model weights
- `../saved_models/emotion_model_final.pt`: Final model weights
- `../saved_models/emotion_model_metadata.json`: Model metadata (feature names, class mappings)

### 4. `predict_emotion.py`
Inference module for predicting emotion vectors.

**Classes:**
- `EmotionPredictor`: Wrapper for inference

**Key methods:**
- `predict(X)`: Predict emotion vectors and labels
- `predict_from_csv(csv_path)`: Predict directly from pose CSV
- `get_emotion_vector_for_movement(csv_path, movement_id)`: Get aggregated emotion vector for a movement

**Functions:**
- `batch_predict()`: Process multiple CSV files

**Usage:**
```python
from predict_emotion import EmotionPredictor

predictor = EmotionPredictor(
    model_path="../saved_models/emotion_model_best.pt",
    metadata_path="../saved_models/emotion_model_metadata.json"
)

predictions = predictor.predict_from_csv("../pose_results_L2/gBR_sBM_c01_d04_mBR0_ch01_pose.csv")

print(predictions['emotion_vector'])  # (num_frames, 5)
print(predictions['emotion_primary'])  # List of emotion labels
print(predictions['memory_tag'])  # List of memory tags
```

### 5. `rerank_matches.py`
Integration with Layer 1: Rerank body-similar matches by emotional similarity.

**Key functions:**
- `cosine_similarity(a, b)`: Cosine distance between emotion vectors
- `euclidean_similarity(a, b)`: Euclidean distance-based similarity
- `rerank_by_emotion()`: Main reranking function
- `rerank_with_confidence()`: Confidence-aware reranking

**Usage:**
```python
from rerank_matches import rerank_by_emotion
import numpy as np

# Layer 1 output
layer1_matches = [
    {"movement_id": "move_045", "body_similarity": 0.92},
    {"movement_id": "move_102", "body_similarity": 0.88},
]

# Query emotion vector (5D)
query_emotion = np.array([0.4, 0.7, 0.8, 0.2, 0.3])

# Archive emotion vectors
archive_emotions = {
    "move_045": np.array([0.5, 0.8, 0.7, 0.3, 0.2]),
    "move_102": np.array([0.8, 0.9, 0.2, 0.1, 0.9]),
}

# Rerank (60% body, 40% emotion)
reranked = rerank_by_emotion(
    layer1_matches,
    query_emotion,
    archive_emotions,
    alpha=0.6,
    similarity_metric='cosine'
)

for match in reranked:
    print(f"{match['movement_id']}: {match['final_score']:.3f}")
```

## Training

### Step 1: Install Dependencies
```bash
pip install torch numpy pandas scikit-learn
```

### Step 2: Train Model
```bash
cd ml_layer2
python train_emotion_model.py
```

**Training configuration (edit in `train_emotion_model.py`)**:
```python
train_model(
    csv_dir="../pose_results_L2",
    epochs=100,
    batch_size=16,
    learning_rate=0.001,
    output_dir="../saved_models",
    use_gpu=True
)
```

**Expected output:**
```
Using device: cuda
Loading and processing data...
Processing gBR_sBM_c01_d04_mBR0_ch01_pose.csv...
Processing gHO_sBM_c01_d19_mHO0_ch01_pose.csv...
Processing gJB_sBM_c01_d07_mJB0_ch01_pose.csv...
Combined dataset shape: (38300, 25)
...
Epoch 10, Loss: 0.1234
...
Training completed!
Best validation loss: 0.0987
```

## Inference

### Single CSV File
```python
from predict_emotion import EmotionPredictor

predictor = EmotionPredictor(
    model_path="../saved_models/emotion_model_best.pt",
    metadata_path="../saved_models/emotion_model_metadata.json"
)

predictions = predictor.predict_from_csv("../pose_results_L2/gBR_sBM_c01_d04_mBR0_ch01_pose.csv")

print(f"Emotion vectors: {predictions['emotion_vector'].shape}")
print(f"Sample: {predictions['emotion_vector'][0]}")
# Output: [0.45 0.72 0.31 0.68 0.52]
#         (emotion_intensity, arousal, tension, weight, flow)
```

### Batch Processing
```python
from predict_emotion import batch_predict
from pathlib import Path

csv_files = list(Path("../pose_results_L2").glob("*.csv"))

predictions = batch_predict(
    csv_files,
    model_path="../saved_models/emotion_model_best.pt",
    metadata_path="../saved_models/emotion_model_metadata.json",
    device='cpu'
)
```

## Layer 1 + Layer 2 Integration

Complete example of matching movements:

```python
import numpy as np
from predict_emotion import EmotionPredictor
from rerank_matches import rerank_by_emotion

# Initialize emotion predictor
predictor = EmotionPredictor(
    "../saved_models/emotion_model_best.pt",
    "../saved_models/emotion_model_metadata.json"
)

# Get emotion vector for query movement
query_emotions = predictor.predict_from_csv("query_pose.csv")
query_emotion_vector = np.mean(query_emotions['emotion_vector'], axis=0)

# Get emotion vectors for archive
archive_emotions = predictor.predict_from_csv("archive_pose.csv")

# Build archive dictionary
archive_emotion_dict = {
    movement_id: np.mean(archive_emotions['emotion_vector'][i:i+1], axis=0)
    for i, movement_id in enumerate(archive_emotions['movement_ids'])
}

# Layer 1 results (from body similarity)
layer1_matches = [
    {"movement_id": "move_045", "body_similarity": 0.92},
    {"movement_id": "move_102", "body_similarity": 0.88},
    {"movement_id": "move_087", "body_similarity": 0.85},
    {"movement_id": "move_211", "body_similarity": 0.82},
    {"movement_id": "move_034", "body_similarity": 0.79},
]

# Rerank by emotion
final_matches = rerank_by_emotion(
    layer1_matches,
    query_emotion_vector,
    archive_emotion_dict,
    alpha=0.6  # 60% body, 40% emotion
)

# Results
for i, match in enumerate(final_matches, 1):
    print(f"{i}. {match['movement_id']}")
    print(f"   Body: {match['body_similarity']:.3f}")
    print(f"   Emotion: {match['emotion_similarity']:.3f}")
    print(f"   Final Score: {match['final_score']:.3f}")
```

## Output Format

### Emotion Vector (5D)
```
[emotion_intensity, arousal, tension, weight, flow]
[0.45,              0.72,     0.31,    0.68,  0.52]
```

Each value ranges from 0 to 1 (normalized by Sigmoid in the model).

### Reranked Matches
```python
[
    {
        "movement_id": "move_045",
        "body_similarity": 0.92,
        "emotion_similarity": 0.89,
        "final_score": 0.907
    },
    {
        "movement_id": "move_087",
        "body_similarity": 0.85,
        "emotion_similarity": 0.86,
        "final_score": 0.854
    },
    ...
]
```

## Hyperparameters

### Training
- **Learning rate**: 0.001
- **Batch size**: 16
- **Epochs**: 100 (with early stopping at patience=20)
- **Optimizer**: Adam
- **LR scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)

### Model Architecture
 64 neurons
 (5/n_classes) neurons
- **Activation**: ReLU
- **Dropout**: 0.3 (encoder), 0.2, 0.2, 0.1 (decreasing)
- **Regression output**: Sigmoid (normalized to [0, 1])

### Reranking
- **alpha (weighting)**: 0.6 (60% body similarity, 40% emotion)
- **Similarity metric**: Cosine or Euclidean distance

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `batch_size=8`
- Use CPU: `device='cpu'`

### Low Validation Loss
- Reduce dropout rates
- Increase learning rate
- Check data quality

### Poor Emotion Predictions
- Ensure CSV files have all required columns (emotion_primary, emotion_intensity, etc.)
- Check feature scaling (features should be normalized)
- Increase training epochs
