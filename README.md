# Emotion Detection from Uploaded Images

## Problem Statement
To develop, design, implement, and optimize a complete solution that integrates machine learning, computer vision, and user interface design to classify emotions from uploaded images.

## Objective
The goal of this project is to develop a **Streamlit-based application** that enables users to upload an image, which will then be processed to detect the emotion of the person in the image using **Convolutional Neural Networks (CNNs).**

## Business Use Cases
- **Healthcare**: Mental Health Monitoring and Support
- **Education**: Personalized Learning and Engagement
- **Customer Service**: Enhancing User Experience
- **Market Research**: Understanding Consumer Sentiment
- **Human Resources**: Improving Employee Engagement and Well-being

## Methodology, Models Used, and Evaluation Results

### Step 1: CNN Model Development

#### Dataset
**FER-2013**: A dataset containing facial images categorized into seven emotions: *Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.*

#### Facial Feature Extraction
**Dlib** is used for facial landmark detection (68 points), enhancing emotion classification accuracy by focusing on key facial features (e.g., eyes, mouth, eyebrows).

#### Model Used: EmotionCNN – Custom Model
##### Architecture:
- **Convolutional Layers**: Feature extraction from input images (64, 128, 256 filters).
- **MaxPooling Layers**: Reduce spatial dimensions to prevent overfitting.
- **Flatten**: Converts extracted features into a 1D vector.
- **Fully Connected Layers**: Dense layers to classify seven emotion categories.
- **Dropout**: Regularization to avoid overfitting.

##### Model Workflow:
1. Input Image (48x48 grayscale)
2. Passed through five convolutional layers with increasing filters
3. Pooling layers reduce dimensions
4. Flatten operation
5. Fully connected layers with softmax output
6. **Output**: Emotion classification

#### Model Performance Before and After Facial Landmarks
| Model     | Train Accuracy (Before) | Test Accuracy (Before) | Train Accuracy (After) | Test Accuracy (After) |
|-----------|------------------------|------------------------|------------------------|------------------------|
| EmotionCNN | 65.62% | 63.03% | 69.35% | 65.54% |
| LeNet      | 47.56% | 44.83% | 48.35% | 45.39% |
| AlexNet    | 42.48% | 43.87% | 40.48% | 44.52% |
| MobileNet  | 41.99% | 44.15% | 42.03% | 43.91% |

### Step 2: Streamlit User Interface
A **web application** allows users to upload an image, process it through the trained CNN model, and display the detected emotion.

#### Key Functionalities:
- **Model Definition**: Implements EmotionCNN with five convolutional layers.
- **Pre-Trained Model Loading**: Loads `best_model_emotion.pth` for inference.
- **Face Detection (Dlib)**: Identifies faces and extracts 68 facial landmarks.
- **Image Preprocessing**: Converts images to grayscale, resizes to (48x48), and applies necessary transformations.
- **User Interface (Streamlit)**: Upload an image in **JPG, PNG** format, display uploaded image, and detect emotions.

#### Use Case Scenarios:
- **Single Face Detected**: Emotion prediction displayed.
- **No Face Detected**: Error message prompts the user to retry or continue.
- **Multiple Faces Detected**: Provides a message stating multiple faces detected.
- **Invalid File Upload**: Restricts to JPG/PNG formats.

## Ethical Considerations
- **Privacy & Security**: Ensuring user data protection and compliance with regulations.
- **Bias & Fairness**: Training models on diverse datasets to avoid misclassifications.
- **Accuracy Concerns**: Potential misclassification risks in sensitive areas like healthcare.
- **Ethical Use**: Avoiding unethical applications such as surveillance or emotional manipulation.

## Conclusion
Emotion detection using CNNs and Streamlit is a powerful tool with applications in multiple industries, including **healthcare, education, and customer experience.** With proper ethical guidelines, it can be used responsibly to enhance human interactions and decision-making processes.
