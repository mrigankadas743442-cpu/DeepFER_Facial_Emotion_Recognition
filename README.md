# 😊 DeepFER - Facial Emotion Recognition using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)

A Deep Learning based Facial Emotion Recognition System capable of predicting **7 human emotions** from facial images and performing **real-time emotion detection** using a webcam.

</div>

---

# 📌 Project Overview

Human emotions play an important role in communication and decision-making. This project presents a Convolutional Neural Network (CNN) based facial emotion recognition system that classifies facial expressions into seven different emotions.

The model has been trained using grayscale facial images of size **48×48** and is capable of recognizing emotions from both uploaded images and live webcam input.

---

# 🎯 Problem Statement

Understanding human emotions automatically is a challenging task in Artificial Intelligence and Computer Vision.

The objective of this project is to build a deep learning model that can accurately classify facial expressions into different emotional categories. Such systems can be useful in:

- Human-Computer Interaction
- Healthcare
- Education
- Smart Surveillance
- Customer Behaviour Analysis
- Mental Health Monitoring

---

# 🚀 Features

✅ Emotion Prediction from Uploaded Images

✅ Real-Time Webcam Emotion Detection

✅ Detects Largest Face Automatically

✅ Predicts 7 Different Emotions

✅ Displays Confidence Score

✅ Saves Captured Images with Prediction

✅ Face Detection using Haar Cascade

✅ Deep Learning CNN Model

✅ TensorFlow SavedModel Support

---

# 😊 Supported Emotions

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😄
- Neutral 😐
- Sad 😢
- Surprise 😲

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | CNN Model |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |
| Seaborn | Confusion Matrix |
| Scikit-learn | Model Evaluation |
| Google Colab | Model Training |
| VS Code | Webcam Application |

---

# 🧠 Model Architecture

The CNN architecture consists of:

- Data Augmentation
- Convolution Layers
- Batch Normalization
- Max Pooling
- Dropout Layers
- Dense Layers
- Softmax Output Layer

Input Image Size:

```
48 × 48 × 1
```

Output Classes:

```
7 Emotion Classes
```
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/895e0e1d-d0e6-428c-aa38-94a70fc2d07d" />

---

# 📊 Model Performance

## Training Results

| Metric | Value |
|---------|--------|
| Training Accuracy | 59.56% |
| Validation Accuracy | **62.45%** |
| Test Accuracy | **61.37%** |
| Test Loss | **1.0084** |

---

# 📈 Classification Report

| Emotion | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Angry | 0.53 | 0.55 | 0.54 |
| Disgust | 0.44 | 0.29 | 0.35 |
| Fear | 0.49 | 0.27 | 0.35 |
| Happy | **0.84** | **0.86** | **0.85** |
| Neutral | 0.52 | 0.65 | 0.58 |
| Sad | 0.46 | 0.52 | 0.49 |
| Surprise | 0.75 | 0.72 | 0.73 |

Overall Test Accuracy:

> **61.37%**

---

# 📷 Emotion Prediction from Uploaded Image

The model accepts a facial image, preprocesses it into grayscale (48×48), and predicts the corresponding emotion with a confidence score.

Example:
<img width="319" height="427" alt="image" src="https://github.com/user-attachments/assets/ee821aca-bff5-49f6-853b-376c0d2a8b30" />

```
Prediction:
Happy

Confidence:
47.16%
```

---

# 🎥 Real-Time Webcam Emotion Detection

The webcam application allows users to:

- Detect face in real time
- Capture image by pressing **SPACE**
- Predict emotion instantly
- Display confidence score
- Save predicted image
- Resume camera by pressing **R**
- Exit by pressing **Q**

---


# 📸 Results

The project provides:

- Accuracy Graph
 <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/b3a7101f-510e-4ba2-a28d-88b2ee1667b1" />

- Loss Graph
 <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/0881b724-c534-4980-89e4-b32e961a169c" />

- Confusion Matrix
 <img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/253b40fd-6e24-45a6-b11e-69662b294624" />

- Real-Time Webcam Prediction
<img width="1280" height="720" alt="capture_6" src="https://github.com/user-attachments/assets/615445a4-6cdc-4a4b-a521-121c8d36430b" />

---

# 🔮 Future Improvements

- Mobile Deployment
- TensorFlow Lite Model
- Face Tracking
- Multiple Face Detection
- Emotion History Graph
- Voice-Based Emotion Recognition
- Flask Web Application
- Streamlit Dashboard

---

# 👨‍💻 Author

**Mriganka Das**

MCA Graduate | AI & Data Science Enthusiast

GitHub:
https://github.com/mrigankadas743442-cpu

