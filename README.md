# Plant Disease Classification Model Report

## 1. Introduction

This report presents the implementation, training, and evaluation of a deep learning model for plant disease classification using convolutional neural networks (CNNs). The model was trained on an image dataset with 38 plant disease classes, utilizing PyTorch and TensorBoard for performance tracking.

## 2. Model Architecture

The `DiseaseClassifier` model is a custom CNN inspired by AlexNet, with additional batch normalization layers for better stability. The model consists of:

### Feature Extraction Layers:

- Five convolutional layers with ReLU activation.
- Batch normalization to improve training speed and stability.
- Max pooling to reduce spatial dimensions.

### Fully Connected Layers:

- Adaptive average pooling for size reduction.
- Two linear layers with dropout for regularization.
- Softmax activation for classification.

## 3. Training and Hyperparameters

### Dataset:

- Image dataset containing 38 plant disease classes.

### Image Preprocessing:

- Resized to 224x224 pixels.
- Data augmentation: horizontal flip, rotation, color jitter, affine transformation.
- Normalization using ImageNet mean and standard deviation.

### Training Details:

- **Loss Function:** CrossEntropyLoss with label smoothing (0.1).
- **Optimizer:** Adam with an initial learning rate of 0.0005.
- **Gradient Clipping:** Applied to prevent exploding gradients (max norm = 1.0).
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor = 0.1, patience = 3).
- **Early Stopping:** Triggered if validation loss does not improve for 4 epochs.
- **Training Duration:** Model trained for up to 20 epochs.

## 4. Performance Metrics

The model was evaluated using the following metrics:

- **Accuracy:** Overall classification correctness.
- **Precision:** Proportion of correctly predicted positive samples.
- **Recall:** Proportion of actual positive samples correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

### 4.1 Training Results

| Epoch | Loss   | Accuracy | Precision | Recall  | F1-score |
|-------|--------|-----------|------------|---------|-----------|
| 1     | 2102.033  | 0.578291  | 0.571974  | 0.578572  | 0.573891  |
| 2     | 1450.64   | 0.802361  | 0.801116  | 0.802642  | 0.801612  |
| ...   | ...      | ...       | ...       | ...      | ...       |
| 20    | 845.0408  | 0.987709  | 0.987698  | 0.98766   | 0.987674  |

### 4.2 Validation Results

| Epoch | Loss   | Accuracy | Precision | Recall  | F1-score |
|-------|--------|-----------|------------|---------|-----------|
| 1     | 465.0744  | 0.711416  | 0.795022  | 0.710892  | 0.699968  |
| 2     | 373.6571  | 0.856192  | 0.876477  | 0.85678   | 0.856917  |
| ...   | ...      | ...       | ...       | ...      | ...       |
| 20    | 202.6924  | 0.990781  | 0.990911  | 0.99068   | 0.990672  |

## 5. Use Case

This plant disease classification model can be applied in various real-world agricultural scenarios, including:

- **Smart Farming:** Assisting farmers in early disease detection to prevent crop loss.
- **Precision Agriculture:** Automating plant disease diagnosis for targeted pesticide use and optimal yield.
- **Agricultural Research:** Providing a tool for researchers to analyze plant disease trends.
- **Mobile & IoT Integration:** Deploying the model in mobile applications for on-field disease detection using smartphone cameras.

## 6. Conclusion

The Plant Disease Classification Model achieved promising performance on the dataset, leveraging a custom CNN architecture with advanced regularization techniques. Future improvements could include:

- Exploring transfer learning with a pretrained model (e.g., ResNet, EfficientNet).
- Increasing dataset size and diversity for better generalization.
- Hyperparameter tuning for further optimization.
