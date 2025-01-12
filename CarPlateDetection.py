import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import pytesseract
import easyocr
import math

IMG_DATA = "dataset/{}/images"
LABELA_DATA = "dataset/{}/labels"

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

reader = easyocr.Reader(['en'])

def preprocess_bbox(bbox_data, img_height, img_width):
    
    bbox_data = bbox_data.strip('\n')
    # class, bbox center x, bbox center y, h, w
    _, x, y, w, h = map(float, bbox_data.split(" "))
    x1 = int((x - w / 2) * img_width)
    x2 = int((x + w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    y2 = int((y + h / 2) * img_height)
    
    return [x1, y1, x2, y2]   


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    if lines is None:
        return 0.0
    angle = 0.0

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))

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

def plateDetection(model, img, data, imgPath):

    results = model.predict(source=imgPath)
    img_h, img_w, _ = img.shape

    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        class_ids = boxes.cls.astype(int)

    

        for xyxy, class_id in zip(xyxys, class_ids):
            print("Class id: ", class_id)
            bbox1 = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

            cv2.rectangle(
                img,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                (36, 255, 12),
                3,
            )

            best_iou = 0
            best_idx = -1
            for d in data:
                [x1, y1, x2, y2] = preprocess_bbox(d, img_h, img_w)
                iou = calculate_iou(bbox1, [x1, y1, x2 - x1, y2 - y1])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = data.index(d)

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (36, 255, 12),
                        3,
                    )

            print("Best iou: ", best_iou)

            #cv2.imshow("image", img)
            #cv2.waitKey(0)

            return xyxys # [x1, y1, x2, y2]

            

def plot_labeled_data(mode='test', model=None):

    fig = plt.figure(figsize=(20, 20)) 
    rows = 4
    columns = 4
    
    imgs_list = os.listdir(IMG_DATA.format(mode))
    random.shuffle(imgs_list)

    #labels_list = os.listdir(LABELA_DATA.format(mode))
    for i, img_name in enumerate(imgs_list[:16]):

        imgPath = os.path.join(IMG_DATA.format(mode), img_name)

        img = cv2.imread(imgPath)
        
        fl = open(os.path.join(LABELA_DATA.format(mode), img_name[:-3] + 'txt'), 'r')

        data = fl.readlines()

        platesCoords = plateDetection(model, img, data, imgPath)

        fl.close()

        if platesCoords is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for plateCoord in platesCoords:

            plateCoord = [int(plateCoord[0]), int(plateCoord[1]), int(plateCoord[2]), int(plateCoord[3])]

            cv2.rectangle(
                img,
                (plateCoord[0], plateCoord[1]),
                (plateCoord[2], plateCoord[3]),
                (36, 255, 12),
                3,
            )

            carPlate = img[plateCoord[1]:plateCoord[3], plateCoord[0]:plateCoord[2]]

            carPlate = deskew(carPlate)

            img_gray = cv2.cvtColor(carPlate, cv2.COLOR_RGB2GRAY) 

            sharpened_image = cv2.filter2D(img_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) )

            #ocr_result = pytesseract.image_to_string(gray_img, config='--psm 7')

            result = reader.readtext(sharpened_image, detail = 0)

        ax = fig.add_subplot(rows, columns, i+1) 
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        

        #ax.set_title(f"Plate {i+1}\n Predicted: {ocr_result.strip()}")
        ax.set_title(f"Plate {i+1}\n Predicted: {result}")

        plt.imshow(sharpened_image, cmap='gray')
        
    plt.show()
    
if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    #results = model.train(data="Lamine.yaml", epochs=50, imgsz=640, device=0)
    plot_labeled_data(model=model)