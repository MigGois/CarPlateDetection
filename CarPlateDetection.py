import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from ultralytics import YOLO
import kagglehub

model = YOLO("yolo11n.pt")

results = model.train(data="coco8.yaml", epochs=100, imgsz=640)


""" # Download latest version
path = kagglehub.dataset_download("andrewmvd/car-plate-detection")

print("Path to dataset files:", path) """