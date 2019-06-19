import cv2
import matplotlib.pyplot as plt
from threading import Thread
import datetime
import time
import argparse
import numpy as np
from sklearn.spatial import distance as dist
import imutils
import SciPy
import playsound
import dlib
from imutils.video import VideoStream
from imutils import face_utils


def sound_alarm(path):
    playsound.playsound(path)
    
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    EAR = (A + B)/ (2.0 * C)

    return EAR


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type = str, default = "", help="path to alarm .WAV file")
#ap.add_argument("-w", "--webcam", required=True, help="index to webcam on the system")

args = vars(ap.parse_args())

#defining 2 constants, one for the EYE_AR_THRESH, and the other for the consecutive no. of frames the if the EYE_AR_CONSEC_FRAMES 
#for which the eye aspect ratio remains below the threshold value.

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 50

#initialize the frame counter and as well as a BOOLEAN used to indicate if the alarm is going off.
COUNTER = 0
ALARM_ON = False

#Initializing dlib's HOG based facial landmark predictor.
print("[INFO] loading facial landmark predictor from dlib's library.")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape-predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right eye"]


print("[INFO] loading the webcam ...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    #grab the FRAME from the threaded video stream file and resize it, and onvert it into grayscale.
    frame = vs.fread()
    frame = imutils.resize(frame, width = 550)
    gray = cv2.cvtColor(frame, cv2.COLORBGR2GRAY)
    
    #detect faces in the grayscale frame.
    rects = detector(gray, 0)
    
    for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            #extract the left and the right eye coordinates, then calculating Eye Aspect Ratio
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            #average the EAR.
            EAR = (leftEAR +rightEAR) / 2.0
            
            
            #compute the convexhull for the left and the right eye, then visualizing each of the eyes.
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)
            

            if EAR < EYE_AR_THRESH:
                COUNTER+=1
                
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        
                        #checking to see if an alarm file was supplied, and if so, start the alarm playing in the background.
                        
                        if args["alarm"]!= "":
                            t = Thread(target=sound_alarm(args["alarm"]))
                            t.deamon = True
                            t.start()
                            
                    cv2.putText(frame, "DROWSINESS ALERT!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            
            
            else:
                COUNTER = 0
                ALARM_ON = False
                            
                    
            
            #draw the eye aspect ratio for sufficient info for debugging.
            cv2.putText(frame, "EAR: {.2f}".format(EAR), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow("FRAME", frame)
        key = cv2.waitKey('q')& 0xFF
        if key == ord('q'):
		break
            
cv2.destroyAllWindows()
vs.stop()
