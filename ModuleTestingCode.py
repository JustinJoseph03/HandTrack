import cv2
import mediapipe as mp
import time
import HandTrackModule as htm

prevTime = 0
currTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw = True) #False hide base landmarks
    lmList = detector.findPosition(img,draw = False)  #False hides custom lm circles
    if len(lmList) != 0:
        print(lmList[4])

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (25, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
