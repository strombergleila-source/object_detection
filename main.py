
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


""" 
# Create and train the model
model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640
) 
"""
# Run the model on the test images
""" model = YOLO("best.pt")
model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.25 
)"""

#model.val(data="dataset/data.yaml")
""" model = YOLO("best.pt")
model.val(data="dataset/data.yaml", plots=True) """

# test the model with the webcam
# model = YOLO("best.pt")
# model.predict(source=0, show=True)


