
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import yaml
import os

# Dosya yolları
model_path = "best.pt"
data_yaml_path = "data.yaml"

# Model ve sınıf isimlerini yükle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}.")

model = YOLO(model_path)

# Sınıf isimlerini data.yaml'den al
with open(data_yaml_path, 'r') as stream:
    data = yaml.safe_load(stream)
class_names = data['names']

def run_inference(image: Image.Image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model.predict(img_bgr, conf=0.25, iou=0.5)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = class_names[cls]
        detections.append([name, conf, x1, y1, x2, y2])

        # Kutuları çiz
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(img_bgr, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(img_rgb)
    df = pd.DataFrame(detections, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
    return output_img, df

# Gradio arayüzü
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Image(type="pil", label="Detected Image"),
        gr.Dataframe(label="Detections")
    ],
    title="YOLOv8 Face Detector",
    description="Upload an image to detect faces using a custom-trained YOLOv8 model."
)

demo.launch()
