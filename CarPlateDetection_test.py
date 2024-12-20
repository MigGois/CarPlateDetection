import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract
import easyocr

#IMG_DATA = "dataset/train/images/01a916bfaa3a7427.jpg"
#LABELA_DATA = "dataset/train/labels/01a916bfaa3a7427.txt"

IMG_DATA = "dataset/test/images/fe6139d150a3e2a8.jpg"
LABELA_DATA = "dataset/test/labels/fe6139d150a3e2a8.txt"
reader = easyocr.Reader(['en'])

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

def preprocess_bbox(bbox_data, img_height, img_width):
    
    bbox_data = bbox_data.strip('\n')
    # class, bbox center x, bbox center y, h, w
    _, x, y, w, h = map(float, bbox_data.split(" "))
    x1 = int((x - w / 2) * img_width)
    x2 = int((x + w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    y2 = int((y + h / 2) * img_height)
    
    return [x1, y1, x2, y2]   


def plot_labeled_data(mode='train'):
    fig = plt.figure(figsize=(20, 20)) 
    rows = 4
    columns = 4
    img = cv2.imread(IMG_DATA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    with open(LABELA_DATA, 'r') as fl:
        data = fl.readlines()

    plot_index = 1

    for d in data:
        bbox = preprocess_bbox(d, img_h, img_w)
        
        carPlate = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        carPlate = cv2.resize(carPlate, (150, 50), interpolation=cv2.INTER_CUBIC)

        img_plate = cv2.resize(carPlate, None, fx = 3, fy = 3, interpolation= cv2.INTER_CUBIC)

        img_gray = cv2.cvtColor(img_plate, cv2.COLOR_RGB2GRAY) 

        blur = cv2.GaussianBlur(img_gray, (3,3), 0)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
   
        ocr_result = pytesseract.image_to_string(clean_thresh, config='--psm 6')

        result = reader.readtext(thresh, detail = 0)

        ax = fig.add_subplot(rows, columns, plot_index)
        ax.imshow(thresh, cmap='gray')
        
        ax.set_title(f"Plate {plot_index}\n Predicted: {ocr_result.strip()}")
        ax.set_title(f"Plate {plot_index}\n Predicted: {result}")
        ax.axis('off')

        plot_index += 1

    plt.show()

    
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    #results = model.train(data="Lamine.yaml", epochs=10, imgsz=640, device=0)
    plot_labeled_data()