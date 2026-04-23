# Driver Drowsiness Detection System

A real-time AI-powered Driver Drowsiness Detection System that monitors the driver's face through a webcam and predicts whether the driver is **Alert (Safe)** or **Drowsy (Unsafe)**.

This project is designed to help reduce road accidents caused by fatigue, sleepiness, or inattentive driving.

# Project Overview

Driver drowsiness is one of the major causes of road accidents worldwide.  
This project uses **Computer Vision** to detect drowsiness in real time using a webcam.

The system captures live video, detects the driver's face, preprocesses the image, and sends it to a trained model for prediction.

# Objectives
- Detect driver drowsiness in real time.
- Improve road safety.
- Use AI for accident prevention.
- Create an easy-to-use monitoring system.

# Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- Streamlit
- NumPy
- Haar Cascade Face Detection

# Model Used
This project uses a **Pre-trained MobileNetV2 Model**.

## Why MobileNetV2?
- Lightweight and fast
- High accuracy
- Optimized for real-time applications
- Suitable for low-resource systems

## Transfer Learning Used
We used **Transfer Learning**, where MobileNetV2 was already trained on millions of images, and then fine-tuned for:
- Alert Driver
- Drowsy Driver

# Project Structure
```bash
DriverDrowsiness/
│── app.py                  # Streamlit UI version
│── main.py                 # Raw OpenCV version (without UI)
│── model.weights.h5        # Trained model weights
│── requirements.txt
│── README.md
│── venv/
