import cv2
import numpy as np
import time
import os
import HandTrackModule as htm

######
brushSize = 15
eraserSize = 100

#####


folderPath = "Header"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []

for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

#print(len(overlayList))
header = overlayList[0]
drawColor = (255,0,255)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionConfidence=0.85)
xprev, yprev = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    #Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)   #flips webcam image

    #Find hand landmarks
    img = detector.findHands(img, draw=False)
    landmarkList = detector.findPosition(img, draw=False)

    if len(landmarkList) != 0:
        #print(landmarkList)

        #tip of index and middle fingers
        x1, y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[12][1:]

        #Check which fingers are up

        fingers = detector.fingersUp()
        #print(fingers)

        #If Selection Mode - two fingers up
        if fingers[1] and fingers[2]:
            xprev, yprev = 0, 0
            print("Selection Mode")
            # Checking for click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20),
                          drawColor, cv2.FILLED)

        #If Drawing Mode - index finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xprev == 0 and yprev ==0:
                xprev, yprev = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xprev, yprev), (x1, y1), drawColor, eraserSize)
                cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, eraserSize)

            else:
                cv2.line(img, (xprev, yprev),(x1,y1), drawColor, brushSize)
                cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, brushSize)

            xprev, yprev = x1, y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2. cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #Setting header image
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
