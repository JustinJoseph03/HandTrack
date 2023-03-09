import cv2
import time
import os
import HandTrackModule as htm

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)   #3 is the width of capture
cap.set(4, hCam)   #4 is the height of capture

folderPath = "fingers"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    # print (f"{folderPath}/{imPath}")
    overlayList.append(image)

print(len(overlayList))
prevTime = 0

detector = htm.handDetector(detectionConfidence=0.75)

#mediapipe fingertip values
tipIds = [4,8,12,16,20]

while True:

    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    #print(landmarkList)

    if len(landmarkList) != 0:
        fingers = []
        #Thumb
        if landmarkList[tipIds[0]][1] > landmarkList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #comparing y coord of tip of finger to its' center (excluding thumb),
        #CV orientation y increases as it goes downward
        for id in range(1,5):
            if landmarkList[tipIds[id]][2] < landmarkList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h,w,c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20,225), (170, 425), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255,0,0), 25)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (400,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)