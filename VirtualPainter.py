import cv2
import numpy as np
import os
import HandTracking as htm

brushThickness = 8
eraserThickness = 50

folderPath = "UI"
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in os.listdir(folderPath)]
header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            if y1 < 125:
                if 350 < x1 < 440:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 530 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 700 < x1 < 820:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 880 < x1 < 1000:
                    header = overlayList[3]
                    drawColor = (0, 255, 255)
                elif 1050 < x1 < 1170:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

        if fingers[1] and not fingers[2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

        if all(fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:137, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
