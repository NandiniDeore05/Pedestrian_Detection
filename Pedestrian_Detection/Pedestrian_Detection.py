# PEDISTRIAN DETECTION

import cv2
import time 
import numpy as np

# CREATING BODY CLASSIFIER
body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# INITIATING VIDEO CAPTURING FROM A VIDEO
cam = cv2.VideoCapture('walking.avi')

while cam.isOpened():
    time.sleep(.1)
    
    # READING THE FRAME
    _,frame = cam.read()
    frame = cv2.resize(frame,None, fx=0.5 , fy=0.5 , interpolation = cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    # PASSING GRAY FRAME TO BODY CLASSIFIER
    bodies = body_classifier.detectMultiScale(image=gray, scaleFactor=1.2, minNeighbors=3)
    
    # EXTRACTING BOUNDING BOXES FOR ANY BODIES IDENTIFIED
    for(x,y,w,h) in bodies:
        cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w , y+h), color=(0,255,255), thickness=2)
        cv2.imshow('PEDISTRIAN DETECTION' , frame)
    
    if cv2.waitKey(1)==13:
        break

cam.release()
cv2.destroyAllWindows()
    