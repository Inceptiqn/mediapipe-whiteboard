import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 8
eraserThickness = 50


folderPath = "UI"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
header = overlayList[0]
drawColor = (0, 0, 255)

# Set Lower Camera Resolution to Reduce Processed Pixels
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Reduce Width (640 pixels)
cap.set(4, 360)  # Reduce Height (480 pixels)

# Hand Tracking
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0

# Canvas to Draw
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # Read and Flip Image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))  # Scale back for consistent UI

    # Detect Hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Process if Landmarks Found
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]  # Index Finger Tip
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle Finger Tip

        # Check which fingers are up
        fingers = detector.fingersUp()

        # Selection Mode
        if fingers[1] and fingers[2]:  # If both index and middle fingers are up
            print("Selection Mode")
            if y1 < 125:  # Check if in header region
                if 350 < x1 < 440:  # Red
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 530 < x1 < 650:  # Blue
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 700 < x1 < 820:  # Green
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 880 < x1 < 1000:  # Yellow
                    header = overlayList[3]
                    drawColor = (0, 255, 255)
                elif 1050 < x1 < 1170:  # Eraser (Black)
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

        # Drawing Mode
        if fingers[1] and not fingers[2]:  # If only the index finger is up
            # Reset starting position if not continuous drawing
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
        
            if drawColor == (0, 0, 0):  # Eraser
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:  # Brush
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        
            xp, yp = x1, y1  # Update the previous position
        else:
            xp, yp = 0, 0  # Reset position when not in drawing mode

        # Clear Canvas
        if all(x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Combine Canvas and Image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Set Header Image
    img[0:137, 0:1280] = header

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
