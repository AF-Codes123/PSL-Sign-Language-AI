# 🤟 AI-Powered Pakistan Sign Language Recognition System

An AI-powered computer vision project that recognizes static Pakistan Sign Language (PSL) hand signs from images using a Convolutional Neural Network (CNN).

## 📌 Project Overview

Communication can be difficult when people who use sign language and people who do not understand sign language cannot communicate easily.

This project explores how Deep Learning and Computer Vision can be used to recognize Pakistan Sign Language hand signs from images.

The system takes a hand-sign image as input and uses a trained CNN model to predict the corresponding PSL sign along with its confidence score.

## 🎯 Objective

The main objectives of this project are:

- Recognize Pakistan Sign Language hand signs from images
- Apply CNN-based image classification
- Understand the complete deep learning workflow
- Build a practical Computer Vision application
- Provide a foundation for future real-time sign language recognition

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- NumPy
- Scikit-learn
- Pandas
- Matplotlib
- Google Colab
- Streamlit

## 📊 Dataset

The project uses the **UAlpha40 Pakistan Sign Language Urdu Alphabet dataset**.

The current implementation uses **36 static PSL sign classes** with available images from the dataset.

Some classes in the original dataset did not contain usable images, so they were not included in the current model.

## 🔄 Project Workflow

```text
PSL Image Dataset
        ↓
Image Preprocessing
        ↓
Train / Validation / Test Split
        ↓
Data Augmentation
        ↓
CNN Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Saved CNN Model
        ↓
Streamlit Application
        ↓
Upload Hand-Sign Image
        ↓
Predicted PSL Sign