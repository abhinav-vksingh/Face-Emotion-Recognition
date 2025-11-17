# Face-Emotion-Recognition
Facial Emotion Recognition (FER) system using MobileNetV2 and the FER2013 dataset. The model classifies seven emotions and supports real-time webcam detection using OpenCV. With transfer learning, fine-tuning, and class weighting, it achieves 70.58% accuracy and runs efficiently for practical use.


Facial Emotion Recognition (FER) – MobileNetV2
A deep-learning project for real-time Facial Emotion Recognition using MobileNetV2, TensorFlow, and OpenCV.

📌 Overview
This project detects seven human emotions—angry, disgust, fear, happy, neutral, sad, surprise—from facial images.
It uses transfer learning with MobileNetV2 and supports real-time webcam emotion detection.

🧠 Model Highlights
Base Model: MobileNetV2 (ImageNet pretrained)
Image Size: 224×224
Training Strategy: Transfer learning + fine-tuning
Accuracy Achieved: 70.58%
Macro F1 Score: 55%
Frameworks: TensorFlow / Keras, OpenCV

📂 Project Structure
├── train.py
├── test.py
├── webcam_demo.py
├── class_indices.json
├── models/
├── data/
└── confusion_matrix.png

📦 Dataset Download
The dataset used for training this project (FER images organized into emotion folders) can be downloaded from the release section.

🔗 Download Dataset:
👉 [Download data.zip](https://github.com/abhinav-vksingh/Face-Emotion-Recognition/releases/download/Dataset-Face-emotion-recognition/data.zip)

Note: Extract the ZIP file into a data/ folder with the following structure:
data/
 ├── train/
 ├── val/
 └── test/

🚀 Training
python train.py --data_dir ./data --model_dir ./models --use_augmentation
🧪 Testing & Evaluation
python test.py --data_dir ./data --model_dir ./models

🎥 Real-Time Demo
python webcam_demo.py --model ./models/fer_mobilenetv2_ft_best.h5 --classes ./models/class_indices.json


🔧 Tech Stack
Python
TensorFlow / Keras
MobileNetV2
OpenCV
NumPy, Matplotlib

🔮 Future Enhancements
Better detectors (RetinaFace / MTCNN)
TensorFlow Lite deployment
GUI for user-friendly interaction
