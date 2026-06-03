

## Overview

This project implements a real-time dance pose matching and feedback system using MediaPipe for pose detection and machine learning for similarity analysis. The system compares input dance poses against a database of known poses and provides real-time feedback on pose correctness and emotional/style similarity.

### Project Goals
- Extract human pose landmarks from video using MediaPipe (33 points)
- Match poses based on body position similarity (Layer 1 - Body Similarity)
- Classify poses based on emotional/style content (Layer 2 - Emotional Similarity)
- Provide real-time feedback to dancers via web interface
- Enable batch and real-time pose matching modes

---


## Pipeline Architecture

```
STEP1: Dataset Preparation
 1.1 Download Videos
 1.2 Extract Frames
 1.3 Convert to Stick Figures

STEP2: Pose Estimation (MediaPipe)
 Detect 33-point body landmarks
 Export as CSV files

STEP3: Machine Learning Models
 Layer 1: Body Similarity (Pose Matching)
 3.1.1 Pose Matching Engine  
 3.1.2 Feature Extraction/Inference  
 3.1.3 Neural Network Architecture  
 Layer 2: Emotional Similarity
 3.2.1 Emotion Model Architecture   
 3.2.2 Feature Extraction   
 3.2.3 Emotion Prediction   

STEP4: Backend API Server
 FastAPI REST + WebSocket endpoints

STEP5: Frontend Ver1 (Static/Batch)
 React batch pose matching interface

STEP6: Frontend Ver2 (Real-time Interactive)
 React real-time UI with live feedback
 WebSocket service for live inference
```

### STEP 1: Dataset Preparation

**Purpose:** Acquire and prepare raw video data for analysis

#### 1.1 Download Videos
- **File:** `download_videos.py`
- **Function:** Fetches video files from URLs specified in CSV
- **Input:** `refined_2M_sBM_one_ch01_ch10_per_genre.csv`
- **Output:** `downloaded_videos/` folder
- **Usage:** `python download_videos.py`

#### 1.2 Extract Frames
- **File:** `extract_frames_to_images.py`
- **Function:** Converts video files into individual frame images
- **Input:** `downloaded_videos/*.mp4`
- **Output:** `extracted_frames/` subfolders with JPG frames
- **Usage:** `python extract_frames_to_images.py`

#### 1.3 Convert to Stick Figures
- **File:** `csv_to_stick_figure_images.py`
- **Function:** Renders pose landmarks as visual stick figure diagrams
- **Input:** `pose_results/` CSV files (from STEP 2)
- **Output:** `stick_images2/` folder with PNG visualizations
- **Usage:** `python csv_to_stick_figure_images.py`

### STEP 2: Pose Estimation

**Purpose:** Extract human body landmarks from video frames using MediaPipe

#### Main Script
- **File:** `run_pose_estimation_folder.py`
- **Function:** Detects 33-point body landmarks using MediaPipe's PoseLandmarker
- **Input:** `extracted_frames/` folder with frame images
- **Output:** `pose_results/` folder with CSV and JSON files
- **Landmarks:** Head, shoulders, elbows, wrists, hip, knees, ankles with (x, y, z, confidence)
- **Usage:** `python run_pose_estimation_folder.py`

#### Core Module
- **File:** `pose_landmarker.py`
- **Function:** Wrapper for MediaPipe pose detection
- **Used by:** `run_pose_estimation_folder.py`, `backend.py`

### STEP 3: Machine Learning Models

**Purpose:** Train models for pose similarity and emotion/style classification

#### Layer 1: Body Similarity (Pose Matching)

**Goal:** Find poses visually similar in body position

##### 3.1.1 Pose Matching Engine
- **File:** `models/ml_layer1/match_pose.py`
- **Purpose:** Compares query poses against archive using embeddings
- **Methods:** Cosine similarity, L2 distance, Visibility filtering
- **Used by:** `backend.py`, `realtime_service/`

##### 3.1.2 Feature Extraction / Pose Inference
- **File:** `models/ml_layer1/pose_inference.py`
 fixed-size embedding vector
- **Model:** `saved_models/pose_encoder.pt`
- **Used by:** `match_pose.py`

##### 3.1.3 Neural Network Architecture
- **File:** `models/ml_layer1/pose_model.py`
- **Architecture:** Multi-layer perceptron
- **Input:** 33 landmarks  (x, y, z, confidence) = 132 values
- **Output:** Fixed-size embedding vector

##### Training & Utilities
- `train_pose_encoder.py` - Train the pose encoder
- `build_archive_embeddings.py` - Precompute archive embeddings
- `evaluate_pose_encoder.py` - Test model performance
- `plot_loss_curve.py` - Visualize training progress

#### Layer 2: Emotional Similarity (Style Matching)

**Goal:** Find poses with similar emotional/style content

##### 3.2.1 Emotion Model Architecture
- **File:** `models/ml_layer2/emotion_model.py`
- **Purpose:** Neural network for emotion/style classification
- **Input:** Pose sequence features
- **Output:** Emotion/style prediction

##### 3.2.2 Feature Extraction
- **File:** `models/ml_layer2/extract_emotion_features.py`
- **Purpose:** Extract features from pose sequences
- **Handles:** Sequence padding, temporal alignment, normalization

##### 3.2.3 Emotion Prediction
- **File:** `models/ml_layer2/predict_emotion.py`
- **Purpose:** Real-time emotion inference
- **Model:** `saved_models/emotion_model.pt`
- **Used by:** `backend.py`, `realtime_service/`

##### Training & Utilities
- `train_emotion_model.py` - Train the emotion classifier
- `rerank_matches.py` - Rerank matches by emotional similarity

#### Integration
- **File:** `models/layer1_layer2_integration.py`
- **Purpose:** Coordinates Layer 1 and Layer 2 for cascade matching
 Final ranking

### STEP 4: Backend API Server

**Purpose:** Expose machine learning models through REST/WebSocket APIs

#### Main Server
- **File:** `backend.py`
- **Framework:** FastAPI (Python)
- **Port:** 8000
- **Endpoints:**
  - `POST /match_pose` - Upload image, get similar poses
  - `GET /archive` - Retrieve archive database
  - `POST /archive_query` - Query archive with filters
  - WebSocket routes to realtime_service

#### Integrations
 pose conversion
- **STEP 3.1:** Uses `match_pose.py` for similarity matching
- **STEP 3.2:** Uses `predict_emotion.py` for emotion filtering
- **STEP 5:** Serves static files for `frontend/`
- **STEP 6:** Routes WebSocket connections to `realtime_service/`

### STEP 5: Frontend Ver1 (Static/Batch)

**Purpose:** Simple web interface for batch pose matching

- **Folder:** `frontend/`
- **Framework:** React
- **Features:** Upload image, click "Match Poses", view ranked similar poses, filter by score/type/emotion
- **API Integration:** Calls `backend.py` REST endpoints

### STEP 6: Frontend Ver2 (Real-time Interactive)

**Purpose:** Advanced web interface with real-time feedback

#### React Frontend
- **Folder:** `frontend2/`
- **Framework:** React
- **Features:** Live webcam capture, real-time pose overlay, live feedback, emotion analysis, movement suggestions

#### WebSocket Service (Python Backend)
- **Folder:** `realtime_service/`
- **Communication:** WebSocket (bidirectional, real-time)

##### Components

###### websocket_routes.py (Entry Point)
- Handles WebSocket connections from `frontend2/`
- Routes incoming pose streams to processors
- Sends back predictions and feedback

###### feature_extractor.py
- Extracts emotion features from live pose sequence
- Buffers poses for temporal analysis

###### motion_buffer.py
- Maintains sliding window of recent poses
- Handles temporal synchronization
- Provides pose history for trend analysis

###### next_movement_predictor.py
- Predicts next recommended movements
- Provides corrective feedback
- Uses STEP 3.1 (body similarity) + STEP 3.2 (emotion)
- Suggests: "your next pose should be X"
- Corrects: "your Y landmark is off, adjust here"

###### archive_interface.py
- Real-time database lookup
- Fast retrieval of similar poses from archive

###### schemas.py
- Data structures for WebSocket messages
- Request/response validation

---

## File Organization

```
.
 README.md (this file - combined documentation)

 STEP1: Data Preparation
 download_videos.py (1.1)   
 extract_frames_to_images.py (1.2)   
 csv_to_stick_figure_images.py (1.3)   

 STEP2: Pose Estimation
 run_pose_estimation_folder.py   
 pose_landmarker.py   

 STEP3: ML Models
 models/   
 __init__.py      
 layer1_layer2_integration.py      
 ml_layer1/ (Body Similarity)      
 match_pose.py (3.1.1)         
 pose_inference.py (3.1.2)         
 pose_model.py (3.1.3)         
 train_pose_encoder.py         
 build_archive_embeddings.py         
 evaluate_pose_encoder.py         
 plot_loss_curve.py         
 saved_models/ (models stored here)         
 test_archive/         
 ml_layer2/ (Emotional Similarity)      
 emotion_model.py (3.2.1)          
 extract_emotion_features.py (3.2.2)          
 predict_emotion.py (3.2.3)          
 train_emotion_model.py          
 rerank_matches.py          
 saved_models/          
 __init__.py          

 STEP4: Backend API
 backend.py   

 STEP5: Frontend Ver1
 frontend/   

 STEP6: Frontend Ver2 + Real-time Service
 frontend2/   
 realtime_service/   
 __init__.py       
 realtime/       
 __init__.py           
 websocket_routes.py           
 feature_extractor.py           
 motion_buffer.py           
 next_movement_predictor.py           
 archive_interface.py           
 schemas.py           

 Data Folders
 downloaded_videos/ (raw videos)   
 extracted_frames/ (video frames)   
 pose_results/ (pose CSV files)   
 stick_images/ & stick_images2/ (visualizations)   
 pose_results_L2/ (ML results)   

 Configuration
 refined_2M_sBM_one_ch01_ch10_per_genre.csv    
```

---

## Component Details

### All Python Files Listed

#### STEP1 Files
- `download_videos.py` - **1.1: Download Videos** - Fetches video files from URLs
- `extract_frames_to_images.py` - **1.2: Extract Frames** - Extracts frames from videos
- `csv_to_stick_figure_images.py` - **1.3: Stick Figures** - Renders pose as stick figures

#### STEP2 Files
- `run_pose_estimation_folder.py` - **Main Pipeline** - Pose estimation using MediaPipe
- `pose_landmarker.py` - **Core Module** - MediaPipe wrapper

#### STEP3 Files (15 total)

Layer 1 - Body Similarity:
- `models/ml_layer1/match_pose.py` - **3.1.1: Pose Matching** - Find visually similar poses
 embeddings
- `models/ml_layer1/pose_model.py` - **3.1.3: Architecture** - Neural network design
- `models/ml_layer1/train_pose_encoder.py` - **Training** - Train the encoder
- `models/ml_layer1/build_archive_embeddings.py` - **Utility** - Precompute embeddings
- `models/ml_layer1/evaluate_pose_encoder.py` - **Utility** - Test performance
- `models/ml_layer1/plot_loss_curve.py` - **Utility** - Visualize training

Layer 2 - Emotional Similarity:
- `models/ml_layer2/emotion_model.py` - **3.2.1: Architecture** - Neural network for emotion
- `models/ml_layer2/extract_emotion_features.py` - **3.2.2: Features** - Extract from sequences
- `models/ml_layer2/predict_emotion.py` - **3.2.3: Prediction** - Real-time emotion inference
- `models/ml_layer2/train_emotion_model.py` - **Training** - Train emotion classifier
- `models/ml_layer2/rerank_matches.py` - **Utility** - Rerank by emotion

Integration:
- `models/layer1_layer2_integration.py` - **Integration** - Coordinate both layers
- `models/__init__.py` - **Package Init** - Export interfaces
- `models/ml_layer2/__init__.py` - **Package Init** - Layer 2 module

#### STEP4 Files
- `backend.py` - **Backend API** - FastAPI server with REST + WebSocket

#### STEP6 Files (8 total)
- `realtime_service/__init__.py` - **Package Init**
- `realtime_service/realtime/__init__.py` - **Sub-package Init**
- `realtime_service/realtime/websocket_routes.py` - **WebSocket Handler** - Live connection management
- `realtime_service/realtime/feature_extractor.py` - **Feature Extraction** - Process live poses
- `realtime_service/realtime/motion_buffer.py` - **Motion Buffering** - Sliding window of poses
- `realtime_service/realtime/next_movement_predictor.py` - **Prediction** - Suggest next moves
- `realtime_service/realtime/archive_interface.py` - **Database Interface** - Fast pose lookup
- `realtime_service/realtime/schemas.py` - **Data Schemas** - Request/response structures

---

## Data Flow

```
STEP1: Raw Data Acquisition
 download_videos.py
  Input: CSV with URLs
  Output: downloaded_videos/ (MP4 files)

 extract_frames_to_images.py
  Input: downloaded_videos/ MP4s
  Output: extracted_frames/ JPG frames

 csv_to_stick_figure_images.py
   Input: pose_results/ CSVs
   Output: stick_images2/ visualizations

        

STEP2: Pose Extraction
 run_pose_estimation_folder.py
  Uses: pose_landmarker.py (MediaPipe)
  Input: extracted_frames/ JPGs
  Output: pose_results/ CSVs

 pose_landmarker.py
   (MediaPipe wrapper - 33 landmarks)

        

STEP3: ML Model Training (Offline)
 Layer 1: Body Similarity
 train_pose_encoder.py  
  Inputs: pose_results/ CSVs  
  Output: saved_models/pose_encoder.pt  
  
 pose_model.py  
  Neural network architecture  
  
 pose_inference.py  
  Uses: pose_encoder.pt  
 embeddings
  
 match_pose.py  
     Uses: pose_inference.py
     Outputs: similarity scores

 Layer 2: Emotional Similarity
 train_emotion_model.py  
  Inputs: pose sequences  
  Output: saved_models/emotion_model.pt  
  
 emotion_model.py  
  Neural network architecture  
  
 extract_emotion_features.py  
  Extract features from sequences  
  
 predict_emotion.py  
     Uses: emotion_model.pt
     Outputs: emotion predictions

 build_archive_embeddings.py
   Pre-compute archive embeddings
   for fast lookup

        

STEP4: Backend Server
 backend.py
 Loads: pose_encoder.pt + emotion_model.pt   
 Imports: match_pose.py, predict_emotion.py   
 Serves: REST API endpoints   
  - /match_pose   
  - /archive   
  - etc.   
 realtime_service

        

STEP5 & STEP6: Frontends
 STEP5: frontend/
  REST API client
  Batch pose matching UI

 STEP6: frontend2/ + realtime_service/
   WebSocket connection
   Real-time feedback
   Live pose matching
```

---

## Running the System

### 1. Data Preparation (One-time setup)
```bash
# Download videos from CSV
python download_videos.py

# Extract individual frames from videos
python extract_frames_to_images.py

# (Skip csv_to_stick_figure_images.py - runs after pose estimation)
```

### 2. Pose Estimation (One-time per dataset)
```bash
# Extract poses from all frames
python run_pose_estimation_folder.py

# Now you can generate stick figures
python csv_to_stick_figure_images.py
```

### 3. ML Model Training
```bash
# Train Layer 1 (Body Similarity)
cd models/ml_layer1
python train_pose_encoder.py       # Creates pose_encoder.pt
python build_archive_embeddings.py # Pre-computes embeddings
python evaluate_pose_encoder.py    # (Optional) Test performance

# Train Layer 2 (Emotional Similarity)
cd ../ml_layer2
python train_emotion_model.py      # Creates emotion_model.pt
```

### 4. Start Backend Server
```bash
# Run FastAPI server on http://localhost:8000
python backend.py

# Check API: http://localhost:8000/docs
```

### 5. Run Frontends (in separate terminals)

**Option A: Static/Batch UI**
```bash
cd frontend
npm install  # (first time only)
npm start    # Runs on http://localhost:3000
```

**Option B: Real-time Interactive UI**
```bash
cd frontend2
npm install  # (first time only)
npm start    # Runs on http://localhost:3001
# (Backend must be running)
```

---

## Understanding the Code

### Documentation in Every File

Every Python file has a docstring at the top explaining:

```python
"""
STEP[N]: [Component Name]
 [Sub-step A]
 [Sub-step B] (THIS FILE)
 [Sub-step C]

PURPOSE: What this module does
Input: Where data comes from
Output: Where results go
Used by: Which files use this
"""
```

### How to Read a File's Docstring
```bash
# See the docstring of any file
head -15 [filename].py

# Examples:
head -15 backend.py
head -15 models/ml_layer1/match_pose.py
head -15 realtime_service/realtime/websocket_routes.py
```

### Understanding Dependencies

Each file tells you:
1. **Which STEP** - Maps to the 6-step pipeline
2. **What it does** - Purpose and functionality
3. **Input/Output** - Data flow
4. **Used by** - Which files depend on it
5. **Uses** - Which files it depends on

### Tracing Data Through the Pipeline

1. Start with `download_videos.py` (STEP 1.1)
2. Follow to `extract_frames_to_images.py` (STEP 1.2)
3. Then `run_pose_estimation_folder.py` (STEP 2)
4. Then to `models/ml_layer1/train_pose_encoder.py` (STEP 3.1 training)
5. Then to `match_pose.py` (STEP 3.1 inference)
6. Finally to `backend.py` (STEP 4)

### Finding Specific Functionality

 `models/ml_layer1/match_pose.py`
 `models/ml_layer1/pose_inference.py` or `models/ml_layer2/extract_emotion_features.py`
 `models/ml_layer2/predict_emotion.py`
 `backend.py`
 `realtime_service/realtime/websocket_routes.py`

---

## Documentation Examples

### Example 1: File from STEP1 (extract_frames_to_images.py)
```python
"""
STEP1: Dataset Preparation
 1.1: Download Videos
 1.2: Extract Frames (THIS FILE)
 1.3: Convert to Stick Figures

PURPOSE: Extracts individual frames from downloaded video files and saves them as images.
Input: downloaded_videos/ folder
Output: extracted_frames/ folder with frame images
"""
```

### Example 2: File from STEP3.1 (match_pose.py)
```python
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
```

### Example 3: File from STEP4 (backend.py)
```python
"""
STEP4: Backend API (THIS FILE)
 Exposes REST endpoints for pose matching and emotion prediction
 Integrates STEP3 ML models (body similarity + emotional similarity)
 Serves STEP5 & STEP6 Frontend applications
 Provides WebSocket routes for real-time predictions

PURPOSE: FastAPI backend server that:
- Loads pose matching model (STEP3.1 - Body Similarity)
- Handles image uploads for pose extraction
- Performs pose matching against archive database
- Integrates emotion prediction (STEP3.2 - Emotional Similarity)
- Returns ranked dance pose matches with visualizations
"""
```

### Example 4: File from STEP6 (websocket_routes.py)
```python
"""
STEP6: Frontend ver2 - Real-time Service
 WebSocket Routes (THIS FILE)
 Feature extraction from live poses
 Real-time prediction and feedback
 Motion buffering and sequence analysis

PURPOSE: FastAPI WebSocket router for real-time interaction.
Handles live pose streams from the frontend and provides real-time pose matching + emotion feedback.
Integrates STEP3 ML models for live inference.
"""
```

---



### Pose Embedding
- Raw pose: 33 landmarks  4 values (x, y, z, confidence) = 132-dimensional vector
 64D (or similar) fixed embedding
- Why: Enables efficient similarity comparison and flexible matching

### Similarity Metrics
- **Cosine Similarity:** Angle between embedding vectors (0-1, higher = more similar)
- **L2 Distance:** Euclidean distance (lower = more similar)
- **Visibility Filter:** Only compare landmarks with high confidence

### Emotion/Style Classification
- **Input:** Sequence of 5-10 recent poses (temporal data)
- **Process:** Extract temporal features (movement patterns, speed, dynamics)
- **Output:** Emotion class (happy, intense, gentle, etc.) or style score

### Real-time Processing Pipeline
- **Motion Buffer:** Keeps 10-20 recent poses for analysis
- **Feature Extraction:** Updates emotion features as new poses arrive
- **WebSocket:** Sends feedback to frontend immediately
- **Cascade Matching:** 
  1. Find similar poses by body position (Layer 1)
  2. Refine by emotional similarity (Layer 2)
  3. Return ranked results

### Model Architecture
- **Layer 1 (Body Similarity):** 
  - Input: 33 landmarks (132 values)
  - Architecture: Multi-layer MLP
  - Output: 64D embedding
  - Training: Contrastive learning on pose similarities

- **Layer 2 (Emotional Similarity):**
  - Input: Sequence of pose embeddings + temporal features
  - Architecture: LSTM or Transformer-based
  - Output: Emotion/style class or score
  - Training: Classification on labeled pose sequences
