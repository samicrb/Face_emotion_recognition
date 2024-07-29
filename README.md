# Face Emotion Recognition

This repository contains a project for recognizing emotions from facial images using deep learning. The project includes data preprocessing, model training, and real-time emotion detection from video streams.

## Project Structure

- **data/**: Contains raw and processed data.
- **models/**: Trained models and older versions for reference.
- **notebooks/**: Jupyter notebooks for data analysis and model training.
- **scripts/**: Python scripts for data preprocessing and training.
- **tests/**: Notebooks and scripts for testing and validation.
- **Live_detection/**: Scripts and files for real-time emotion detection.

## Installation

To install the required dependencies, run:


pip install -r requirements.txt

For real-time emotion detection from video, you have to add a video named "exemple_video.mp4" to the root and run:

python emotion_recognition.py

Usage
Data Preprocessing
To preprocess the AffectNet data, run:
python scripts/data_preprocessing_affecnet.py

Training
To train the emotion recognition model, use: