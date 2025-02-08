import cv2 as cv
import numpy as np
import time
import dlib
import pyautogui as pg
from math import hypot
import keyboard


cap = cv.VideoCapture(0)
predictor_path = "D:\Class Materials\IT\Web\LearningRepo01\Class\TGMT\Eye-tracing\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

font = cv.FONT_HERSHEY_SIMPLEX
color = [145, 245, 75]
rightEyePoints = [36, 37, 38, 39, 40, 41]
leftEyePoints = [42, 43, 44, 45, 46, 47]

pg.moveTo(960, 540, duration=0.2)
mx, my = pg.position()


def midPoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def getBlinkingRatio(landmarkPoint):
    leftPoint = (landmarks.part(
        landmarkPoint[0]).x, landmarks.part(landmarkPoint[0]).y)
    rightPoint = (landmarks.part(
        landmarkPoint[3]).x, landmarks.part(landmarkPoint[3]).y)
    topPoint = midPoint(landmarks.part(
        landmarkPoint[1]), landmarks.part(landmarkPoint[2]))
    botPoint = midPoint(landmarks.part(
        landmarkPoint[4]), landmarks.part(landmarkPoint[5]))

    # horLine = cv.line(frame, leftPoint, rightPoint, (0, 255, 0), 2)
    # verLine = cv.line(frame, topPoint, botPoint, (0, 255, 0), 2)
    verLineLength = hypot(
        (topPoint[0] - botPoint[0]), (topPoint[1] - botPoint[1]))
    horLineLength = hypot(
        (leftPoint[0] - rightPoint[0]), (leftPoint[1] - rightPoint[1]))
    ratio = horLineLength/verLineLength

    return ratio

def getGazeRatio(landmarkPoint):
    #seperate eye region from face
    eyeRegion = np.array([(landmarks.part(landmarkPoint[0]).x, landmarks.part(landmarkPoint[0]).y),
                          (landmarks.part(landmarkPoint[1]).x, landmarks.part(landmarkPoint[1]).y),
                          (landmarks.part(landmarkPoint[2]).x, landmarks.part(landmarkPoint[2]).y),
                          (landmarks.part(landmarkPoint[3]).x, landmarks.part(landmarkPoint[3]).y),
                          (landmarks.part(landmarkPoint[4]).x, landmarks.part(landmarkPoint[4]).y),
                          (landmarks.part(landmarkPoint[5]).x, landmarks.part(landmarkPoint[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv.polylines(mask, [eyeRegion], True, 255, 2)
    cv.fillPoly(mask, [eyeRegion], 255)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv.dilate(mask, kernel, 5)
    eye = cv.bitwise_and(gray, gray, mask=mask)
    
    minX = np.min(eyeRegion[:, 0])
    maxX = np.max(eyeRegion[:, 0])
    minY = np.min(eyeRegion[:, 1])
    maxY = np.max(eyeRegion[:, 1])

    eyeGray = eye[minY:maxY, minX:maxX]
    
    _, eyeThreshold = cv.threshold(eyeGray, 70, 255, cv.THRESH_BINARY)
    height, width = eyeThreshold.shape

    eyeLeftThreshold = eyeThreshold[0: height, 0: int(width/2)]
    eyeLeftWhite = cv.countNonZero(eyeLeftThreshold)

    eyeRightThreshold = eyeThreshold[0: height, int(width/2):width]
    eyeRightWhite = cv.countNonZero(eyeRightThreshold)

    eyeTopThreshold = eyeThreshold[0: int(height/2), 0: width]
    eyeTopWhite = cv.countNonZero(eyeTopThreshold)

    eyeBotThreshold = eyeThreshold[int(height/2): height, 0: width]
    eyeBotWhite = cv.countNonZero(eyeBotThreshold)

    if eyeLeftWhite == 0:
        gazeHorRatio = 1
    elif eyeRightWhite == 0:
        gazeHorRatio = 5
    else: gazeHorRatio = eyeLeftWhite/eyeRightWhite

    if eyeTopWhite == 0:
        gazeVerRatio = 1
    elif eyeBotWhite == 0:
        gazeVerRatio = 2
    else: gazeVerRatio = eyeTopWhite/eyeBotWhite

    gazeRatio = [gazeHorRatio, gazeVerRatio]
    return gazeRatio

while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # blinks
        leftBlinkingRatio = getBlinkingRatio(leftEyePoints)
        rightBlinkingRatio = getBlinkingRatio(rightEyePoints)
        ratio = (leftBlinkingRatio + rightBlinkingRatio)/2

        #cv.putText(frame, str(ratio), (50, 150),font, 3, color, 2)
        th = 7  # open 0 → 7, closed 7 → ... lower = more sensitive, numbers may vary by distance, set to 4 for close-up cam

        if ratio > th:
            cv.putText(frame, "LEFT CLICK", (50, 150), font, 3, color, 2)
            pg.click()



        # seperate iris from eye white
        leftEyeGazeRatio = getGazeRatio(leftEyePoints)
        rightEyeGazeRatio = getGazeRatio(rightEyePoints)
        gazeRatio = [((leftEyeGazeRatio[0] + rightEyeGazeRatio[0])/2),
                     ((leftEyeGazeRatio[1] + rightEyeGazeRatio[1])/2)]
        cv.putText(frame, "Hor:" +
                   str(gazeRatio[0]), (50, 50), font, 1, color, 2)
        cv.putText(frame, "Ver:" +
                   str(gazeRatio[1]), (50, 30), font, 1, color, 2)


        
        if gazeRatio[0] < 0.3:
            cv.putText(frame, "LOOKING LEFT", (50, 90), font, 1, color, 2)
            pg.moveTo(mx-15, my, duration=0.05)
            mx -= 15

        elif 0.3 < gazeRatio[0] < 1.2:
            cv.putText(frame, "LOOKING CENTER", (50, 90), font, 1, color, 2)

        else:
            cv.putText(frame, "LOOKING RIGHT", (50, 90), font, 1, color, 2)
            pg.moveTo(mx+15, my, duration=0.05)
            mx += 15

        if gazeRatio[1] < 0.1:
            cv.putText(frame, "LOOKING UP", (50, 150), font, 1, color, 2)
            pg.moveTo(mx-15, my, duration=0.05)
            my -= 15

        elif gazeRatio[1] > 1:
            cv.putText(frame, "LOOKING DOWN", (50, 150), font, 1, color, 2)
            pg.moveTo(mx+15, my, duration=0.05)
            my += 15

    cv.imshow("Frame", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
