#!/usr/bin/env python

'''
TO RUN:
python ex_realtime.py -c pi-drowsiness-detection/haarcascade_frontalface_default.xml  -p shape_predictor_68_face_landmarks.dat
'''

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,
	help="boolean used to indicate if TrafficHat should be used")
ap.add_argument("-u", "--user", type=str, default='driver',
	help="String used for pushing data to server")
args = vars(ap.parse_args())

# driver_id = sys.argv[5]
print args['user']

'''
cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
'''
# Global Setting for if use lights / sound
serial = True
sound = True

# Finds SERIAL
if serial:
    client = LEDDriver(find_serial_port())
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

	# return the eye aspeqct ratio
	return ear

def euclidean_dist(ptA, ptB):
	# compute and return the euclidean distance between the two
	# points
	return np.linalg.norm(ptA - ptB)


vs = VideoStream(src=0).start()         # imutil package

# initialize CV2's face detector (Haar-based) and then create
# the facial landmark predictor, this is in the git repo, needed to download it
shape_predict = "shape_predictor_68_face_landmarks.dat"
detector = cv2.CascadeClassifier(args["cascade"])                   # cv2 Detector is faster via Haar Cascades
predictor = dlib.shape_predictor(args["shape_predictor"])           # DLIB Detector via linear SVM + HOG

# Two threshold variables for interactability
EAR_THRESH = .2
EAR_CONSEC = 3

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print 'Init time is: ', time.time() - t0, ' (s)'

playflag = False
distract = False

# Timing initiations
t_play = time.time()
t_temp = t_play

# Reduces serial command and sound clutter with timers
t_sound = 0
t_serial = 0


i=0

# Loop
while(True):
    # Imutil Video Read frame capture and manipulation
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


    # Prints frame rate
    tl = time.time()
    t_diff = tl-t_temp
    cv2.putText(frame, "FPS {}".format(int(1/(t_diff))), (20,20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    t_temp = tl

	# Turn LEDs off when suitable
    if serial and (not playflag):
        client.off()

    # Reset playflag after time period
    if (tl-t_serial) > 2:
		playflag = False
		# i=0
		# print 'reset'



    # detect faces in the grayscale image
    for (x, y, w, h) in rects:
        i = 0
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        # construct a dlib rectangle object from the Haar cascade bounding box
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


    ## if no faces in frame
    if len(rects) > 0:

        # For checking wakefullness
        eye1 = shape[lStart:lEnd]
        eye2 = shape[rStart:rEnd]

        # print("face")
        ear1 = eye_aspect_ratio(eye1)
        ear2 = eye_aspect_ratio(eye2)
        if (ear1 < EAR_THRESH) and (ear2 < EAR_THRESH):

			# Sets number of iterations till blink -> eyes closed
			# too_long = 10
			t_loop_1 = time.time()
			if serial and (t_loop_1-t_serial) > .5:
				if not playflag:
					client.pulse(Color(0,255,0),3,10)
					t_serial = t_loop_1
				# if (i > too_long) and (i%3 == 0):
				# 	print 'triggd'
				# 	if not playflag:
				# 		client.set(255,Color(0,255,0))
				# 		playflag = True
				# 		t_play = time.time()
				#
				# i += 1

			# Checks sound condition
			if sound and (t_loop_1-t_sound) > 1:
				playsound('beep-02a.mp3',block=False)
				t_sound = t_loop_1
    else:
		# Looking away loop
		t_loop_2 = time.time()
		if serial and (t_loop_2-t_serial) > .5:
			if not playflag:
				client.red()
				t_serial = t_loop_2
				playflag = True
                # t_play = time.time()
		# Checks sound condition 2

		if sound and (t_loop_2-t_sound) > 1:
			playsound('beep-01a.mp3',block=False)
			t_sound = t_loop_2
		i=0

        ## Change color

    #Image process for displaying
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    ## No Face sh
    # time.sleep(0.1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
		break


print('Aborted, runtime:'), time.time()-t0, ' (s)'
if serial: client.close()
# cap.release()
cv2.destroyAllWindows()
vs.stop()
