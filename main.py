import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe
from cvzone.PlotModule import LivePlot

idList = [22, 23, 24, 26,110,157,158,159,160,161,130,243]
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
ploty = LivePlot(640,360,[20,50])
ratioList = []

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img,draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, (255, 0, 255),cv2.FILLED)

        leftup = face[159]
        leftdown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer,_ = detector.findDistance(leftup,leftdown)
        lengthHor,_ = detector.findDistance(leftLeft,leftRight)
        cv2.line(img,leftup,leftdown, (0,200,0),3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = (lengthVer/lengthHor)*100
        ratioList.append(ratio)


        imgplot = ploty.update(ratio)

        cv2.imshow('ImagePlot',imgplot)
        # print(lengthVer)


    cv2.imshow('Image', img)
    cv2.waitKey(25)
