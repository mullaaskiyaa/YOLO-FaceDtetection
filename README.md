YOLO Face Detection using Python

This project implements real-time face detection using the YOLO (You Only Look Once) deep learning model with Python. The system detects human faces accurately in images, videos, and live webcam streams.

🚀 Features

Real-time face detection

Works on:

Images

Video files

Webcam / live camera feed

High accuracy and fast performance using YOLO

Easy to run and modify

🛠️ Technologies Used

Python 3.x

YOLO (YOLOv8 / YOLOv5 – update if needed)

OpenCV

NumPy

Ultralytics (for YOLOv8)

📂 Project Structure
yolov8-face-detector/
│
├── models/
│   └── yolov8-face.pt
│
├── images/
│   └── test.jpg
│
├── videos/
│   └── test.mp4
│
├── face_detect.py
├── requirements.txt
└── README.md

cd yolov8-face-detector


Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate   # Windows


Install dependencies

pip install -r requirements.txt

▶️ Usage
Detect faces in an image
python face_detect.py --image images/test.jpg

Detect faces in a video
python face_detect.py --video videos/test.mp4

Detect faces using webcam
python face_detect.py --camera

📸 Output

Detected faces are highlighted with bounding boxes

Confidence score is displayed for each detected face

📦 requirements.txt (example)
opencv-python
numpy
ultralytics

🧠 How It Works

Input image/video is read using OpenCV

YOLO model processes the frame

Faces are detected and bounding boxes are drawn

Output is displayed in real time

🔮 Future Improvements

Face recognition (identity matching)

Mask detection

Emotion detection

Deployment as a web application
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

DEMO Link : https://e6e31d6768f8dd434b.gradio.live



