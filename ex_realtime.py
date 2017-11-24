#!/usr/bin/env python

import cv2
import numpy as np
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import argparse

from defs import *
import time
from scipy.spatial import distance as dist
import Carbon

# Packages for serial
import serial
import sys
from serial.tools.list_ports import comports as list_comports

# Sound library
from playsound import playsound

# Arduiono LED Library
from LEDDriver import *


# Global Setting for if use lights / sound
serial = True
sound = True

# Finds SERIAL
if serial: client = LEDDriver(find_serial_port())

client.green()

# Start time
t0 = time.time()

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def analyzeImage(im):
    # Read Image
    #im = cv2.imread("capture.jpg");
    size = im.shape         # Functions later use this to calibrate camera

    ################################################################
    # DLIB Example ################################################################
    ################################################################

    # load the input image, resize it, and convert it to grayscale
    # Can change image below
    im = imutils.resize(im, width=500)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    shape = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(im, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(im, (x, y), 1, (0, 0, 255), -1)


    # show the output image with the face detections + facial landmarks
    #cv2.imshow("Output", image)
    #cv2.imwrite('facialFeatures.jpg', image)
    return im, rects, shape

def euclidean_dist(ptA, ptB):
	# compute and return the euclidean distance between the two
	# points
	return np.linalg.norm(ptA - ptB)

import thread

def input_thread(a_list):
    raw_input()
    a_list.append(True)


################################################################
# IMAGE CAPTURE ############################################################
# This is somethign I found online to open the mac webcam and take, save a photo
################################################################
cap = cv2.VideoCapture(0)
# vs = VideoStream(cap).start()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor, this is in the git repo, needed to download it
shape_predict = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier()
predictor = dlib.shape_predictor(shape_predict)
# Gets photo

# Comment out serial such for computer only example
# with serial.Serial(port, 9200) as ser:
ear_thresh = .2


print 'Init time is: ', time.time() - t0, ' (s)'

playflag = False
distract = False
t_play = time.time()
while(True):
    # print 'loop'
    tl = time.time()

    # Reset playflag after time period
    # print tl
    # print t_play
    if (tl-t_play) > 2:
        playflag = False

    # frame = vs.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1,
	# 	minNeighbors=5, minSize=(30, 30),
	# 	flags=cv2.CASCADE_SCALE_IMAGE)

    ret, frame = cap.read()

    #good running
    if serial and (not playflag):
        client.off()

    #analyzeImage(frame)
    # May want to chane to grayscale eventually
    image, faces, shape = analyzeImage(frame)

    ## if no faces in frame
    if len(faces) > 0:

        # For checking wakefullness
        eye1 = shape[36:42]
        eye2 = shape[42:48]

        # print("face")
        ear1 = eye_aspect_ratio(eye1)
        ear2 = eye_aspect_ratio(eye2)
        if (ear1 < ear_thresh) and (ear2 < ear_thresh):
            print("wake Up!")
            if serial:
                if not playflag:
                    print 'yes!'
                    client.pulse(Color(0,255,0),15,15)
                    playflag = True
                    t_play = time.time()
                else:
                    t_play = t_play
            if sound: playsound('beep-01a.mp3',block=False)

    else:
        # print("no face")
        if serial:
            if not playflag:
                client.red()
                playflag = True
                t_play = time.time()
            else:
                t_play = t_play

        ## Change color

    #Image process for displaying
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    ## No Face sh
    # time.sleep(0.1)
    cv2.imshow('frame', rgb)

    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
		break

print ''
print('Aborted, runtime:'), time.time()-t0, ' (s)'
if serial: client.off()
cap.release()
cv2.destroyAllWindows()
