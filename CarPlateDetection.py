import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import kagglehub

model = YOLO("yolo11n.pt")

results = model.train(data="CarPlates/Lamine.yaml", epochs=20, imgsz=640)

