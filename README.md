# Face Mask Detection with Live Alert System
## This project is a real-time face mask detection system that uses computer vision and deep learning to monitor mask compliance. It provides a live video feed, classifies faces as either "with mask" or "without mask," and triggers an audio alert when a person is detected without a mask.

## Features
  Real-time Detection: Analyzes a live webcam feed to detect faces in real-time.

  Deep Learning Model: Utilizes a Convolutional Neural Network (CNN) trained with TensorFlow and Keras for high accuracy.

  Live Alert System: Employs Pygame to play an audio alert when an unmasked face is detected. The sound plays only once per detection to avoid continuous beeping.

  Optimized Performance: Uses OpenCV and Haar Cascades for efficient face detection, ensuring smooth performance on standard hardware.

  Modularity: The project is broken down into three logical scripts: dataset preparation, model training, and live detection.

## Table of Contents
1. Project Structure
2. Prerequisites
3. Installation and Setup
4. Usage
5. Project Walkthrough
6. Deliverables
7.License

## Project Structure
face-mask-detector/
├── dataset/
│   ├── images/
│   ├── annotations/
├── preprocessed_dataset/
│   ├── with_mask/
│   └── without_mask/
├── haarcascade_frontalface_default.xml
├── alert.mp3
├── face_mask_detector_model.h5
├── prepare_dataset.py
├── train_model.py
└── detect_mask_video.py

##  Prerequisites
    pip install opencv-contrib-python
    pip install tensorflow
    pip install keras
    pip install numpy
    pip install imutils
    pip install pygame
    pip install scikit-learn

##  Installation and Setup(Cloing the repository)
    git clone https://www.github.com/BL1183757/Face-Mask-Detector-With-Live-Alert-System
    cd Face-Mask-Detector-With-Live-Alert-System

## Download the dataset:
Download a face mask dataset from Kaggle or another source that includes images and XML annotations.

Place the images and annotations folders inside the dataset directory.

## Download project files:
Download haarcascade_frontalface_default.xml and place it in the project root.

Download a short MP3 file, rename it to alert.mp3, and place it in the project root.

## Usage
Follow these steps to run the project.

## Prepare the dataset:
     python prepare_dataset.py

## Train the model:
    python train_model.py

## Run the live detection system:
    python detect_mask_video.py
Press q to exit the video stream.

## Project Walkthrough
prepare_dataset.py: This script reads each image and its corresponding XML annotation file. It uses the bounding box coordinates from the XML file to crop each face and saves the cropped image into either the preprocessed_dataset/with_mask or preprocessed_dataset/without_mask folder.

train_model.py: This script loads the preprocessed images, converts them into a numerical format, and splits them into training and testing sets. It then builds, compiles, and trains a CNN model to classify the images. Finally, the trained model is saved for later use.

detect_mask_video.py: This script opens a webcam video stream. For each frame, it uses haarcascade_frontalface_default.xml to quickly detect face locations. It then passes the detected face regions to the trained face_mask_detector_model.h5 for classification. Based on the prediction, it draws a colored bounding box and label on the face. An audio alert is triggered using Pygame's mixer when a "No Mask" label is detected.

##  Deliverables
Trained model file (face_mask_detector_model.h5)

Real-time detection script (detect_mask_video.py)

Short video demo showcasing the system's functionality

Project report (in PDF format)

GitHub repository with all project files

## License
This project is open-sourced under the MIT License.
