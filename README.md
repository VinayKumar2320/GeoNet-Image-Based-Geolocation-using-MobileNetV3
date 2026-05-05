# GeoNet-Image-Based-Geolocation-using-MobileNetV3

Overview

GeoNet is a deep learning system designed to estimate geographic location from street-level images. The model combines image classification and regression to predict:

The state/region (classification task)
The latitude and longitude offset relative to a learned centroid (regression task)

This hybrid approach improves localization accuracy while keeping the model efficient and scalable.

Key Features
Multi-task learning: classification + coordinate regression
Lightweight architecture using MobileNetV3
Centroid-based geolocation refinement
Optimized for Apple Silicon (MPS) and GPU acceleration
End-to-end pipeline from training to submission generation

Methodology
1. Data Processing
Images are resized to 224×224
Data augmentation includes:
Horizontal flipping
Color jitter
Labels include:
state_idx → mapped to model classes
Latitude and longitude
2. State-Based Modeling
Each state is mapped to a unique class index
Centroids (mean latitude/longitude) are computed per state
The model predicts:
State probability distribution
Coordinate offset from centroid
3. Model Architecture
Backbone: MobileNetV3 (pretrained)
Head:
Classification layer for state prediction
Regression layer for GPS offset prediction
4. Loss Function

The total loss combines:

Cross-entropy loss for classification
Regression loss for coordinate prediction
Loss = CE(state_pred, state_true) + λ * distance(pred_coords, true_coords)
Training
Requirements

Install dependencies:

pip install torch torchvision torchaudio timm pandas numpy scikit-learn tqdm pillow
Device Support

The code automatically selects:

Apple Silicon GPU (MPS)
NVIDIA GPU (CUDA)
CPU fallback
Training Details
Batch size: 8
Train-validation split: 90/10
Stratified sampling based on state labels
Mixed precision training enabled
Inference
Loads trained weights (geonet_mobilenetv3.pth)
Predicts:
State probabilities
Coordinate offsets
Converts predictions into final latitude/longitude
Outputs submission file in required format
Results

The model produces geolocation predictions by combining:

Coarse localization via classification
Fine-grained refinement via regression

This significantly improves prediction stability compared to direct coordinate regression.

How to Run

Update dataset paths in the notebook:

DATA_ROOT = "path_to_dataset"
Train the model:
Run all cells in sg9152_code.ipynb
Generate predictions:
Ensure model weights are saved
Run inference section
Output:
Submission file (.csv) ready for evaluation
Future Improvements
Use transformer-based vision models (ViT, ConvNeXt)
Add temporal or metadata features if available
Improve regression with geodesic loss functions
Ensemble multiple backbones
Fine-tune on region-specific subsets
Acknowledgements
PyTorch and TorchVision ecosystem
TIMM (PyTorch Image Models) library
Standard geolocation benchmarking practices
