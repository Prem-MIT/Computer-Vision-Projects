import cv2
import time
import os
import HandTrackingModule as htm

#####################################
wCam, hCam = 640, 480
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(detectionCon= 0.75)

#we choose 4, 8, 12, 16, and 20 because it represents tip of fingers respectively
tipIds = [4, 8, 12,16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        fingers = []

    # this is for thumb(open and close)
        if lmList[tipIds[0]][1]> lmList[tipIds[0] - 1][1]:  # Greater than (>) for right hand thumb
                                                            # Less than (<) for left hand thumb
            fingers.append(1)
        else:
            fingers.append(0)

    # this is for remaining fingers (open and close)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print (fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
