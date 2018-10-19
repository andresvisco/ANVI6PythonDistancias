# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:51:44 2018

@author: andres.visco
"""

from scipy.spatial import distance as dist
from imutils import  perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="imagen")
ap.add_argument("-w", "--width", type=float, required=False,
	help="ancho del objeto mas a la izquierda")
args = vars(ap.parse_args())

imagen = "IMG_6098.JPG"
width = 1.258
#width=78.9

image = cv2.imread(imagen)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))
refObj = None

for c in cnts:
    if cv2.contourArea(c)<100:
        continue
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])
    if refObj is None:
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        refObj = (box, (cX, cY), D / width)
        continue
    orig = image.copy()
    
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
    cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 1)
    refCoords = np.vstack([refObj[0], refObj[1]])
    objCoords = np.vstack([box, (cX, cY)])

    
    for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
        cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
        cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
        cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),color, 2)
        
        D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
        distancia = D*2.54
        (mX, mY) = midpoint((xA, yA), (xB, yB))
        cv2.putText(orig, "{:.1f}cm".format(distancia), (int(mX), int(mY - 10))
        ,cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF
        cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
    