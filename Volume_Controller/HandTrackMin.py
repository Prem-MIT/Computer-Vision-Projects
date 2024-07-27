import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)  # will show landmarks when hands are present in camera

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # for converting landmark values to integer
                print(id, cx, cy)

                #if id == 4:
                cv2.circle(img, (cx, cy), 20, (255, 0 ,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)  #cTIme - current time
                             #pTime - previous time
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)