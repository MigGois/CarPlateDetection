import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
from paddleocr import PaddleOCR
import re
import time
from imutils import contours

IMG_DATA = "dataset/test/images/a3ad91fabd188be3.jpg"
BLACKLIST = ['AL', 'AND', 'A', 'BY', 'B', 'BIH', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EST', 'FIN', 'F', 'D', 'GR', 'GB', 'H', 'IS', 'IRL', 'I', 'LV', 'FL', 'LT', 'L', 'M', 'MD', 'MC', 'MNE', 'NL', 'NMK', 'N', 'PL', 'P', 'RO', 'RSM', 'SRB', 'SK', 'SLO', 'E', 'S', 'UA', 'UK', 'V']

ocr = PaddleOCR(use_angle_cls = True, lang='en', show_log=False)


def preprocess_bbox(bbox_data, img_height, img_width):
    
    bbox_data = bbox_data.strip('\n')
    # class, bbox center x, bbox center y, h, w
    _, x, y, w, h = map(float, bbox_data.split(" "))
    x1 = int((x - w / 2) * img_width)
    x2 = int((x + w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    y2 = int((y + h / 2) * img_height)
    
    return [x1, y1, x2, y2]   


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Coordenadas da interseção
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Área da interseção
    inter_area = max(0, x_inter_max - x_inter) * max(0, y_inter_max - y_inter)

    # Área das caixas
    area1 = w1 * h1
    area2 = w2 * h2

    # IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def find_iou(plateCoords, label_path, width, height):

    plateCoords = [plateCoords[0], plateCoords[1], plateCoords[2] - plateCoords[0], plateCoords[3] - plateCoords[1]] 
    
    best_iou = 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:

        bbox = preprocess_bbox(line, height, width)
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        iou = calculate_iou(plateCoords, bbox)

        if iou > best_iou:
            best_iou = iou

    return best_iou


def plateDetection(model, imgPath):

    results = model.predict(source=imgPath)

    boxes = results[0].boxes.cpu().numpy()
    xyxys = boxes.xyxy

    return xyxys # [x1, y1, x2, y2]


    
def post_processing(image):

    """ img_h, img_w, _ = image.shape

    if img_h < 500:
        scale = round(600/img_h)
        image = cv2.resize(image, (img_w * scale, img_h * scale), interpolation = cv2.INTER_LANCZOS4) """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return gray


def image_processing(path):

    processingResults = []

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = img.copy()

    platesCoords = plateDetection(model, path)

    for plateCoord in platesCoords:

        plateCoord = [int(plateCoord[0]), int(plateCoord[1]), int(plateCoord[2]), int(plateCoord[3])]

        cv2.rectangle(
            img1,
            (plateCoord[0], plateCoord[1]),
            (plateCoord[2], plateCoord[3]),
            (36, 255, 12),
            3,
        )

        carPlate = img[plateCoord[1]:plateCoord[3], plateCoord[0]:plateCoord[2]]

        result = post_processing(carPlate)
        
        result1 = ocr.ocr(result, cls=True)

        data = ""

        test = result1[0]

        if test is not None:
            for res in test:
                word = str(res[1][0])
                if not(len(word) <= 3 and word in BLACKLIST):
                    data += str(res[1][0])

        data = re.sub(r'[^a-zA-Z0-9]', '', data)
        
        processingResults.append([result, data, plateCoord])

    return processingResults, img1


def plate_accuracy():

    files = os.listdir("dataset/test/images")

    totalPlates = 0

    foundPlates = 0

    correctPlates = 0

    tp, fp, fn = 0, 0, 0

    start_time = time.time()
    

    for file in files:

        img_path = "dataset/test/images/" + file

        file = file.replace(".jpg", ".txt")

        label_path = "dataset/test/plateLabels/" + file

        loc_plate_path = "dataset/test/labels/" + file

        results, img1 = image_processing(img_path)

        img_h, img_w, _ = img1.shape

        lines = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line != "":
               totalPlates += 1

        for result in results:
            
            foundPlates += 1

            iou = find_iou(result[2], loc_plate_path, img_w, img_h)

            if iou > 0.8:
                tp += 1
            else:
                fp += 1

            for line in lines:

                #print("Resultado Modelo:", result[1].upper())
                #print("Ground Truth:", line.upper().strip())

                if result[1].upper() == line.upper().strip():
                    correctPlates += 1
                    break

    end_time = time.time()

    fn = totalPlates - tp

    elapsed_time = end_time - start_time
    time_image = (elapsed_time/168) * 1000

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cp = round((correctPlates/foundPlates) * 100, 2)


    print("PlateDetection".center(24, "-"))
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1_score * 100:.2f}%")

    print()
    print("TextDetection".center(24, "-"))
    print(f"CorrectPlates: {cp}%")    

    print()
    print("Time".center(24, "-"))
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Time per Image: {time_image:.2f} ms")              


def plot_labeled_data(path):

    fig = plt.figure(figsize=(20, 20)) 
    rows = 4
    columns = 4
    plot_index = 1

    results,orig = image_processing(path)

    for result in results:
        ax = fig.add_subplot(rows, columns, plot_index)

        ax.imshow(result[0], cmap='gray')

        ax.set_title(f"Plate {plot_index}\n Predicted: {result[1]}")
        ax.axis('off')

        plot_index += 1

    ax = fig.add_subplot(rows, columns, plot_index)
    ax.imshow(orig)
    ax.set_title(f"Original Image")

    plt.show()


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    #model = YOLO("yolo11s.pt")
    #results = model.train(data="Lamine.yaml", epochs=100, imgsz=640, device=0)
    #plot_labeled_data("dataset/test/images/d8daed582e6cce2d.jpg")
    plate_accuracy()

