import cv2
import time
import numpy as np
import HandTrackModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#
widthCam, heightCam = 640, 480
#

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)
prevTime = 0

detector = htm.handDetector(detectionConfidence= 0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume.getMute()
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]

vol = 0
barVol = 400
volPercent = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        #print(landmarkList[4], landmarkList[8])

        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 7, (255, 0 ,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 2)
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        #Hand Range = 13 to 190
        #Volume Range = -65 to 0

        vol = np.interp(length, [13, 185], [minVol, maxVol])  #adjust length sensitivity here
        barVol = np.interp(length, [13, 185], [400, 150])  # adjust green bar sensitivity here
        volPercent = np.interp(length, [13, 185], [0, 100])  # adjust volume percentage sensitivity here

        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 19:
            cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (255,0,0), 2)
    cv2.rectangle(img, (50, int(barVol)), (85, 400), (255,0,0), cv2.FILLED)
    cv2.putText(img, f"{int(volPercent)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255,0,0), 2)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (40,50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255,0,0), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)