# Pose Siamese Network - Training & Inference Guide

## Overview

This is a **metric-learning neural network** using **self-supervised learning with triplet loss** to learn pose embeddings. Instead of comparing raw pose keypoints directly, the model converts each pose into a learned embedding vector, enabling efficient similarity matching.

### How It Works

```
User Pose (66-dim) → Neural Network → Embedding (32-dim, normalized)
Archive Pose (66-dim) → Neural Network → Embedding (32-dim, normalized)

Similarity = L2 distance between embeddings (lower = more similar)
```

### Self-Supervised Training Strategy

The model learns using **triplet loss** without requiring manual labels:

- **Anchor**: Original pose (normalized)
- **Positive**: Slightly augmented version of the same pose (+ Gaussian noise)
- **Negative**: Different random pose (normalized)

The loss function pushes anchors close to positives and far from negatives:
```
Loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

---

## Files

### Core Components

1. **`pose_model.py`**
   - `PoseEncoder` class: 3-layer neural network
   - Input: 66-dim pose vectors (33 landmarks × 2 coordinates)
   - Output: 32-dim normalized embeddings
   - Architecture: 66 → 128 → 64 → 32 (with L2 normalization)

2. **`train_pose_encoder.py`**
   - `PoseTripletDataset` class: Loads pose CSVs and creates triplets
   - Data pipeline: CSV → frames → normalized poses → triplets
   - `train()` function: Training loop with Adam optimizer and learning rate scheduling
   - Outputs: Trained model saved to `saved_models/pose_encoder.pt`

### Utilities

3. **`pose_inference.py`**
   - `PoseEmbedder` class: Production-ready inference wrapper
   - Methods:
     - `embed_pose(pose)`: Convert single pose to embedding
     - `embed_poses(poses)`: Convert multiple poses to embeddings
     - `find_similar_poses(query, archive, top_k)`: Find similar poses in archive

4. **`evaluate_pose_encoder.py`**
   - Performance evaluation and statistics
   - Embedding quality metrics
   - Retrieval quality assessment
   - Interactive pose similarity testing

---

## Training

### Setup

```bash
cd /path/to/models/ml_layer1
```

### Run Training

```bash
python3 train_pose_encoder.py
```

#### What Happens During Training

1. **Data Loading**: Loads all CSV files from `pose_results/`
   - Each CSV has rows with (x, y, z) coordinates per landmark
   - Extracts 33 landmarks per frame → 66-dim vectors
   - Filters to only complete frames (all 33 landmarks present)
   - Result: ~15,744 pose frames loaded

2. **Training Process** (50 epochs):
   - **Epoch 1-10**: Loss drops rapidly (0.0184 → 0.0029)
   - **Epoch 10-20**: Fine-tuning with reduced LR (0.0005)
   - **Epoch 20-50**: Further refinement with continued LR decay
   - **Optimization**: Adam optimizer with gradient clipping
   - **Batch size**: 32, Learning rate: 0.001 → 0.00003

3. **Output**: Trained model saved to `saved_models/pose_encoder.pt` (~140 KB)

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input dimension | 66 | 33 landmarks × 2 (x, y) |
| Embedding dimension | 32 | Learned representation size |
| Batch size | 32 | Poses per batch |
| Learning rate | 0.001 | With StepLR decay (γ=0.5 every 10 epochs) |
| Margin (triplet loss) | 0.5 | Distance threshold for negative samples |
| Augmentation noise | σ=0.01 | Small Gaussian perturbation |
| Epochs | 50 | Training iterations |

---

## Evaluation

### Run Evaluation

```bash
python3 evaluate_pose_encoder.py
```

#### Output Includes

1. **Embedding Statistics**
   - Norm check (should all be ~1.0 due to L2 normalization)
   - Effective rank of embedding space

2. **Retrieval Quality**
   - Recall@k metrics (how often nearby frames are in top-k)

3. **Interactive Pose Matching**
   - 5 random query poses
   - Shows top-10 most similar poses and distances
   - Verifies model finds contextually similar poses

#### Expected Behavior

- Similar consecutive frames have **small distances** (0.04-0.15)
- Different movements have **larger distances** (0.2-0.5)
- Embedding norms are all **exactly 1.0** (normalized)

---

## Inference / Usage

### Basic Usage

```python
from pose_inference import PoseEmbedder
import numpy as np

# Initialize embedder
embedder = PoseEmbedder("saved_models/pose_encoder.pt")

# Embed a single pose
pose = np.random.randn(66).astype(np.float32)  # 33 landmarks × 2
embedding = embedder.embed_pose(pose)  # Returns (32,) normalized vector

# Embed multiple poses
poses = np.random.randn(100, 66).astype(np.float32)
embeddings = embedder.embed_poses(poses)  # Returns (100, 32)

# Find similar poses in archive
archive_embeddings = embedder.embed_poses(archive_poses)
results = embedder.find_similar_poses(query_pose, archive_embeddings, top_k=10)

for rank, (archive_idx, distance) in enumerate(results, 1):
    print(f"{rank}. Archive pose #{archive_idx}: distance={distance:.4f}")
```

### Integration with Your Application

```python
# When user uploads a new pose
user_pose = extract_pose_keypoints(user_video)  # Get 66-dim vector

# Find matches in your dance archive
embedder = PoseEmbedder("models/ml_layer1/saved_models/pose_encoder.pt")
archive_embeddings = embedder.embed_poses(all_archive_poses)
matches = embedder.find_similar_poses(user_pose, archive_embeddings, top_k=10)

# Return top matches to user
for rank, (archive_idx, distance) in enumerate(matches, 1):
    print(f"Match {rank}: {archive[archive_idx]['name']} (similarity: {1 - distance:.2%})")
```

---

## Data Format

### Input CSV Format

Each CSV in `pose_results/` has columns:
- `frame_index`: Frame number in the video
- `landmark_id`: Landmark ID (0-32, 33 total)
- `x`, `y`, `z`: 3D coordinates
- Additional metadata: `video_name`, `genre`, `section`, etc.

### Pose Vector Format

```
[x₀, y₀, x₁, y₁, x₂, y₂, ..., x₃₂, y₃₂]  # 66 dimensions
```

Where:
- Landmarks 0-32: Full body keypoints (from MediaPipe or similar)
- x, y: Normalized image coordinates
- z: Depth/confidence

### Normalization

Each pose is normalized to be **translation and scale invariant**:

1. **Centering**: Subtract centroid (mean of all landmarks)
2. **Scaling**: Divide by L2 norm of centered pose

This makes the model robust to:
- Different body sizes
- Different positions in the frame
- Different camera distances

---

## Model Architecture

```
Input: (batch_size, 66)
  ↓
Linear(66 → 128) + ReLU
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 32)
  ↓
L2 Normalization
  ↓
Output: (batch_size, 32) with unit norm
```

### Why This Design

- **Small architecture**: Only 10,656 parameters (efficient)
- **Normalized embeddings**: Enables cosine similarity ≈ L2 distance
- **Moderate dimensions**: 32-dim balances expressiveness vs. efficiency

---

## Loss Curves

Training shows healthy convergence:

```
Epoch 1:   Loss = 0.0184  (initial learning)
Epoch 10:  Loss = 0.0029  (rapid improvement)
Epoch 20:  Loss = 0.0020  (stabilizing)
Epoch 30:  Loss = 0.0016  (fine-tuning)
Epoch 50:  Loss = 0.0012  (final convergence)
```

**Interpretation**: Loss decreases as the model learns to:
- Push augmented poses (positives) closer to anchors
- Push different poses (negatives) farther from anchors

---

## Troubleshooting

### Issue: Low Recall Scores

**Cause**: Model needs more training or better augmentation

**Solutions**:
- Increase training epochs to 100+
- Increase augmentation noise to σ=0.02
- Use different margin values (try 0.3 or 0.7)

### Issue: Embeddings Collapse (all zeros)

**Cause**: Learning rate too high or initialization issues

**Solutions**:
- Reduce learning rate to 0.0005
- Check data normalization (should be centered and scaled)

### Issue: Training is Slow

**Cause**: Large dataset and no GPU

**Solutions**:
- Use GPU (model automatically detects CUDA)
- Reduce dataset size or batch size
- Pre-compute embeddings for archive poses

---

## Future Improvements

1. **Better Augmentation**:
   - Rotation perturbations
   - Partial pose dropping (occlusion simulation)
   - Temporal augmentation (interpolated frames)

2. **Advanced Training**:
   - Hard negative mining
   - Curriculum learning
   - Multi-task learning (add pose classification head)

3. **Model Optimization**:
   - Quantization for mobile deployment
   - Knowledge distillation for faster inference
   - Different embedding dimensions (16-dim for speed)

4. **Evaluation**:
   - Held-out test set evaluation
   - Cross-dancer generalization metrics
   - Real-time performance benchmarks
