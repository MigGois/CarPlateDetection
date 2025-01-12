import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract
import easyocr
import math


IMG_DATA = "dataset/test/images/e57d38ae6a921518.jpg"
LABELA_DATA = "dataset/test/labels/e57d38ae6a921518.txt"

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

            best_iou = 0
            best_idx = -1
            for d in data:
                [x1, y1, x2, y2] = preprocess_bbox(d, img_h, img_w)
                iou = calculate_iou(bbox1, [x1, y1, x2 - x1, y2 - y1])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = data.index(d)

            print("Best iou: ", best_iou)

            #cv2.imshow("image", img)
            #cv2.waitKey(0)

            return xyxys # [x1, y1, x2, y2]

def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,0.02*peri, True)
        print("Poly: ", len(approx))
        print("-------------")
        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
            index = i  # Also save index to contour


    warped = None  # Stores the warped license plate image
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = orig.shape[0]
        width = orig.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)

    if warped is None:
        print("Warped Fail! 😡")
 
    return biggest, imgContour, warped  # Change - also return drawn image


def plot_labeled_data(mode='train'):
    fig = plt.figure(figsize=(20, 20)) 
    rows = 4
    columns = 4
    img = cv2.imread(IMG_DATA)
    img1 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(LABELA_DATA, 'r') as fl:
        data = fl.readlines()

    plot_index = 1

    platesCoords = plateDetection(model, img, data, IMG_DATA)

    for plateCoord in platesCoords:

        plateCoord = [int(plateCoord[0] - 2), int(plateCoord[1] - 2), int(plateCoord[2] + 2), int(plateCoord[3] + 2)]

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

        imgDial = cv2.dilate(imgCanny, kernel,iterations=2)

        imgThres = cv2.erode(imgDial, kernel, iterations=2)

        biggest, imgContour, warped = getContours(imgThres, carPlate)

        img_h, img_w, _ = carPlate.shape

        scale = 1

        if img_h < 720 and warped is None:
            scale = round(720/img_h)
            warped = cv2.resize(carPlate, (img_w * scale, img_h * scale), interpolation = cv2.INTER_LANCZOS4)
            warped = cv2.filter2D(warped, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) )

        img_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY) 

        #sharpened_image = cv2.filter2D(img_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) )

        #ocr_result = pytesseract.image_to_string(img_gray, config='--psm 7')

        result = reader.readtext(img_gray, detail = 0)

        ax = fig.add_subplot(rows, columns, plot_index)
        
        ax.imshow(img_gray, cmap='gray')

        #ax.set_title(f"Plate {plot_index}\n Predicted: {ocr_result.strip()}")
        ax.set_title(f"Plate {plot_index}\n Predicted: {result}")
        ax.axis('off')

        plot_index += 1

    ax = fig.add_subplot(rows, columns, plot_index)
    ax.imshow(img1)
    ax.set_title(f"Original Image")

    plt.show()


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    #model = YOLO("yolo11s.pt")
    #results = model.train(data="Lamine.yaml", epochs=100, imgsz=640, device=0)
    plot_labeled_data()