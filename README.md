# live_detec
Facial Emotion Recognition
This project uses OpenCV, TensorFlow, and Keras to recognize facial emotions in images and videos. The model is trained on the FER-2013 dataset.

Table of Contents
1. Introduction
2. Installation
3. Usage
4. Project Structure
5. Model Creation and Training
6. Video Testing
7. References

Introduction
This project aims to detect facial emotions in real-time from videos. The detected emotions include anger, disgust, fear, happiness, sadness, surprise, and neutrality.

Installation
Clone the repository and install the required dependencies:

bash
Copier le code
git clone https://github.com/samicrb/ros2_ws
cd live_detec/FaceEmotionRecognition
pip install -r requirements.txt
Usage
Data Preprocessing
The data_preprocessing.py script prepares the data for model training.

bash
Copier le code
python3 scripts/data_preprocessing.py
Model Training
The train_model.py script trains the CNN model on the preprocessed data.

bash
Copier le code
python3 scripts/train_model.py
Video Testing
The video_emotion_recognition.py script detects faces in a video and displays the recognized emotion.

bash
Copier le code
python3 scripts/video_emotion_recognition.py --video_path path_to_your_video.mp4
Project Structure
kotlin
Copier le code
.
├── data
│   ├── fer2013.csv
│   ├── test_images.npy
│   ├── test_labels.npy
│   ├── train_images.npy
│   └── train_labels.npy
├── models
│   └── emotion_model.keras
├── notebooks
│   └── data_analysis.ipynb
├── scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── video_emotion_recognition.py
└── README.md
Model Creation and Training
Data Preprocessing: The data_preprocessing.py script reads the data from the FER-2013 CSV file, resizes the images to 48x48 pixels, normalizes them, and saves them as .npy files.

Model Training: The train_model.py script loads the preprocessed data, constructs a CNN model with several layers (Conv2D, MaxPooling2D, Flatten, Dense), and trains the model on the training data. The EarlyStopping and ModelCheckpoint callbacks are used to improve training.

Video Testing
The video_emotion_recognition.py script uses OpenCV to capture frames from the video, detect faces, predict emotions, and display the results on the frame.

Face Detection: Uses cv2.CascadeClassifier to detect faces.
Emotion Prediction: Uses the trained model to predict the emotion for each detected face.
Displaying Results: Displays the detected faces with the predicted emotion.
References
FER-2013 Dataset
OpenCV Documentation
TensorFlow Documentation