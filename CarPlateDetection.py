import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
from paddleocr import PaddleOCR
import re


IMG_DATA = "dataset/test/images/a3ad91fabd188be3.jpg"
LABELA_DATA = "dataset/test/labels/a3ad91fabd188be3.txt"


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


def plateDetection(model, imgPath):

    results = model.predict(source=imgPath)

    boxes = results[0].boxes.cpu().numpy()
    xyxys = boxes.xyxy

    return xyxys # [x1, y1, x2, y2]


def order_points(pts):

    center = np.mean(pts) # Step 1: Find centre of object

    shifted = pts - center # Step 2: Move coordinate system to centre of object

    theta = np.arctan2(shifted[:, 0], shifted[:, 1]) # Step 3: Find angles subtended from centroid to each corner point

    ind = np.argsort(theta) # Step 4: Return vertices ordered by theta

    return pts[ind]


def getContours(img, orig):  
    biggest = np.array([])
    maxArea = 300
    imgContour = orig.copy()  
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,0.02*peri, True)
        

        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
            index = i 


    warped = None  
    if index is not None: 
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) 
        height = orig.shape[0]
        width = orig.shape[1]
        
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        biggest = order_points(src)
        dst = order_points(dst)

        M = cv2.getPerspectiveTransform(src, dst)

        phi = math.atan2(M[2][1], M[2][2])
        psi = math.atan2(M[1][0], M[0][0])

        if math.cos(psi) == 0:
            thetaa = math.atan2(-M[2][0], (M[1][0]/math.sin(psi)))
        else:
            thetaa = math.atan2(-M[2][0], (M[0][0]/math.cos(psi)))

        pi = 22/7

        phid = phi*(180/pi)
        thetaad = thetaa*(180/pi)
        psid = math.degrees(psi)

        M = cv2.getPerspectiveTransform(src, dst)

        if abs(psid) > 35:
            psid = 0
            print("Not Rotating")
            M = np.zeros((3, 3))

        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)
 
    return biggest, imgContour, warped 

    
def post_processing(image):

    img_h, img_w, _ = image.shape

    if img_h < 640:
        scale = round(640/img_h)
        image = cv2.resize(image, (img_w * scale, img_h * scale), interpolation = cv2.INTER_LANCZOS4)

    return image


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

        kernel = np.ones((3,3))

        img_gray = cv2.cvtColor(carPlate, cv2.COLOR_RGB2GRAY) 

        imgBlur = cv2.GaussianBlur(img_gray,(5,5),1)

        imgCanny = cv2.Canny(imgBlur,100,200)

        imgDila = cv2.dilate(imgCanny, kernel,iterations=2)

        imgThres = cv2.erode(imgDila, kernel, iterations=2)

        _, _, warped = getContours(imgThres, carPlate)

        notWarped = False

        if warped is None:
            warped = carPlate.copy()
            notWarped = True


        result = post_processing(warped)
        
        result1 = ocr.ocr(result, cls=True)

        data = ""

        test = result1[0]

        if test is None and not notWarped:
            result = post_processing(carPlate)
            result1 = ocr.ocr(result, cls=True)
            test = result1[0]

        if test is not None:
            for res in test:
                data += str(res[1][0])

        data = re.sub(r'[^a-zA-Z0-9]', '', data)

        
        processingResults.append([result, data])

    return processingResults


def plate_accuracy():
    files = os.listdir("dataset/test/images")

    totalPlates = 0

    foundPlates = 0

    correctPlates = 0

    for file in files:

        img_path = "dataset/test/images/" + file

        file = file.replace(".jpg", ".txt")

        label_path = "dataset/test/plateLabels/" + file

        results = image_processing(img_path)

        lines = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line != "":
               totalPlates += 1

        for result in results:
            
            foundPlates += 1

            for line in lines:

                if result[1].upper() == line.upper().strip():
                    correctPlates += 1

    fp = round((foundPlates/totalPlates) * 100, 2)
    cp = round((correctPlates/foundPlates) * 100, 2)

    print("FoundPlates:", fp, "%")
    print("CorrectPlates:", cp, "%")                    


def plot_labeled_data(mode='train'):

    fig = plt.figure(figsize=(20, 20)) 
    rows = 4
    columns = 4
    plot_index = 1

    results = image_processing(IMG_DATA)

    image = cv2.imread(IMG_DATA)

    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for result in results:
        ax = fig.add_subplot(rows, columns, plot_index)

        ax.imshow(result[0], cmap='gray')

        ax.set_title(f"Plate {plot_index}\n Predicted: {result[1]}")
        ax.axis('off')

        plot_index += 1

    ax = fig.add_subplot(rows, columns, plot_index)
    ax.imshow(image1)
    ax.set_title(f"Original Image")

    plt.show()


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    #model = YOLO("yolo11s.pt")
    #results = model.train(data="Lamine.yaml", epochs=100, imgsz=640, device=0)
    #plot_labeled_data()
    plate_accuracy()

























""" def iou():
    results = model.predict(source=imgPath)
    img_h, img_w, _ = img.shape

    for result in results:

        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        class_ids = boxes.cls.astype(int)

        for xyxy, class_id in zip(xyxys, class_ids):
            print("Class id: ", class_id)
            bbox1 = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

            best_iou = 0
            best_idx = -1
            for d in data:
                [x1, y1, x2, y2] = preprocess_bbox(d, img_h, img_w)
                iou = calculate_iou(bbox1, [x1, y1, x2 - x1, y2 - y1])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = data.index(d)

            print("Best iou: ", best_iou) """