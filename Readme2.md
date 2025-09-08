## Music Genre Classification

### This project is an end-to-end machine learning pipeline for classifying music into genres using audio feature extraction and a deep learning model. It also includes a Streamlit web app that allows users to upload audio files and get real-time predictions.

### Features

  Preprocess audio files into Mel-spectrograms

  Split long audio files into smaller chunks for training

  Train a deep learning model (CNN) for music genre classification

  Save and visualize training history

  Interactive Streamlit app for real-time predictions with audio upload

  Support for multiple audio formats (.wav, .mp3)

### Dataset

The project uses the GTZAN Music Genre Dataset, which contains:

 10 Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

 100 audio files per genre (30 seconds each, .wav)

### Dataset structure:
Data/
 ├── genres_original/
 │   ├── blues/
 │   ├── classical/
 │   ├── country/
 │   ├── disco/
 │   ├── hiphop/
 │   ├── jazz/
 │   ├── metal/
 │   ├── pop/
 │   ├── reggae/
 │   └── rock/

### Preprocessing Steps

Load audio files with Librosa

Split into overlapping chunks (4s with 2s overlap)

Extract Mel-spectrograms

Resize spectrograms into a fixed shape (148, 148)

Convert spectrograms into model-ready NumPy arrays

### Model Architecture

Input: Mel-spectrogram (148x148x1)

Convolutional Neural Networks (CNNs)

Dense layers with softmax output

Optimizer: Adam

Loss: Categorical Crossentropy

### Training

Save training history into training_history.json

Plot accuracy and loss curves

Evaluate model on test data

### Streamlit Web App

The project also includes a Streamlit app for live testing.

#### Run the app locally
    streamlit run app.py

### App Features

 Upload an audio file (.wav or .mp3)

 View waveform and Mel-spectrogram

 Get real-time genre prediction

 (Optional) Extend app to predict mood/tempo

### Installation

#### Clone this repo:
    git clone https://github.com/BL1183757/Internship-Projects.git
    cd Internship-Projects

#### Install dependencies:
        pip install -r requirements.txt

#### Run Jupyter Notebook for training:
        jupyter notebook Music_Genre.ipynb

#### Launch Streamlit app:
        streamlit run app.py

### Requirements

Python 3.8+

TensorFlow / Keras

Librosa

NumPy, Pandas

Matplotlib

scikit-learn

Streamlit

(Optional) pydub, ffmpeg

### Results

Achieved good accuracy on GTZAN dataset

Model generalizes well to unseen songs

### Future Work

Add mood detection (happy, sad, energetic)

Deploy on Streamlit Cloud / Hugging Face Spaces

Enhance dataset with larger music collections

### Author

Developed by Bhavay Khandelwal
Aspiring Data Scientist | ML + AI Enthusiast


