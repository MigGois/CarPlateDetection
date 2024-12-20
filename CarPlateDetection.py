import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import pytesseract
from PIL import Image

IMG_DATA = "dataset/{}/images"
LABELA_DATA = "dataset/{}/labels"
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
    
    imgs_list = os.listdir(IMG_DATA.format(mode))
    random.shuffle(imgs_list)

    labels_list = os.listdir(LABELA_DATA.format(mode))
    for i, img_name in enumerate(imgs_list[:16]):
        print("\nImage " + str(i+1))
        img = cv2.imread(os.path.join(IMG_DATA.format(mode), img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        
        fl = open(os.path.join(LABELA_DATA.format(mode), img_name[:-3] + 'txt'), 'r')
        data = fl.readlines()

        for d in data:

            bbox = preprocess_bbox(d, img_h, img_w)
            img_plate = cv2.rectangle(img=img, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(201, 185, 0), thickness=2)
            carPlate = img_plate[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            img_plate = cv2.resize(carPlate, None, fx = 3, fy = 3, interpolation= cv2.INTER_CUBIC)

            img_gray = cv2.cvtColor(img_plate, cv2.COLOR_RGB2GRAY) 
            blur = cv2.GaussianBlur(img_gray, (5,5), 0)

            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            print(pytesseract.image_to_string(clean_thresh, config = '--psm 6'))
            
        fl.close()
        fig.add_subplot(rows, columns, i+1) 
        plt.imshow(img)
        
    plt.show()
    
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    #results = model.train(data="Lamine.yaml", epochs=10, imgsz=640, device=0)
    plot_labeled_data()