#!/usr/bin/env python
'''
ABOUT:
This is the final project for Interactive Device Design at UCBerkeley, ME290U /
CS 294 - 84. Below is the script to run our computer demo of the real time driver
attention tracking demo. We would like to thanks the following people:
	- Adrian Rosebrock of the PyImageSearch blog
	- Satya Mallick of the LearnOpenCv blog

CODE STRUCTURE: To begin the code, you will see many imports followed by constant
defitions. The program then proceeeds into a frame by frame loop at max possible
frame rate. In this loop, the program shrinks each frame, then looks for faces
and calculates certain features to determine if the user is focused. When the
user is distracted to any level, the program provides feedback. Once initiated,
click on the feedpythoback window and press q to quit.

TO RUN:
python DEMO.py -u MyName
'''
###############################################################################
###############################################################################

# Import functions

# Basic
import numpy as np
import argparse
import time
import datetime
from scipy.spatial import distance as dist

# CV
import cv2
import dlib

# Image Processing and Viewing
import imutils
from imutils.video import VideoStream
from imutils import face_utils

# Packages for Serial
import serial
import sys
from serial.tools.list_ports import comports as list_comports

# Sound library
from playsound import playsound

# Our Functions
from clientdefs import AttenthiaClient			# Web Client commands
from imdefs import *						# Image Processing
from LEDDriver import *						# Arduino Driver

###############################################################################
###############################################################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", type=str, default='driver',
	help="String used for pushing data to server")
args = vars(ap.parse_args())

# driver_id = sys.argv[5]
print('Welcome: ' + args['user'])
###############################################################################

# Global Setting for if use lights / SoundToggle
SerialToggle = False
SoundToggle = False
WebToggle = False

# Finds SerialToggle
if SerialToggle:
    client = LEDDriver(find_SerialToggle_port())

# Hardcodes web client
if WebToggle:
	cli = AttenthiaClient("https://idd-wb-attenthia.herokuapp.com")

# DLIB's SWM + HOG detector and deep learnign predictor
shape_predict_dir = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()						# Dlib detector is slower but more robust
predictor = dlib.shape_predictor(shape_predict_dir)

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

###############################################################################
# Start time
t0 = time.time()
vs = VideoStream(src=0).start()         # imutil package
print('Init time is: ', time.time() - t0, ' (s)')

# Loop to try and calibrate EAR_THRESH
t_cal = 3
calibration_data = []
if SerialToggle: client.blue()
while (time.time()-t0) < t_cal:
	# Read Frame
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)							#Dlib detector
	for (j,rect)in enumerate(rects):

		(x, y, w, h) = rect_to_bb(rect)
		# construct a dlib rectangle object from the Haar cascade bounding box
		rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# For checking wakefullness
		eye1 = shape[lStart:lEnd]
		eye2 = shape[rStart:rEnd]

		# print("face")
		er1 = EyesRatio(eye1)
		er2 = EyesRatio(eye2)

		# Append data into calibration data
		calibration_data.append(er1)
		calibration_data.append(er2)

# Two threshold variables for interactability
if min(calibration_data) > .3:			# If the user responded to the blink prompt
	EAR_THRESH = .2
else:
	EAR_THRESH = min(calibration_data)*1.3
print EAR_THRESH
BOUNDING_BOX_THRESH = 0.1
YAWN_THRESH = .85



###############################################################################



# Timing initiations
t_state = 0			# Timer that stores last time state changed
t_temp = 0			# Temp timer for FPS

# Work with state loops rather than timings
state = ''			# 'clear', 'distract', 'tired'
state_temp = ''
change_time = .1	# Minimum time between changing states ~detection lag

## Image properties for PnP Projection #########################################
size_frame = (253,450,3)
focal_length = size_frame[1]
center = (size_frame[1]/2, size_frame[0]/2)
camera_matrix = np.array(
						 [[focal_length, 0, center[0]],
						 [0, focal_length, center[1]],
						 [0, 0, 1]], dtype = "double"
						 )
# 3D model points. Defined in example
model_points = np.array([
							(0.0, 0.0, 0.0),             # Nose tip
							(0.0, -330.0, -65.0),        # Chin
							(-225.0, 170.0, -135.0),     # Left eye left corner
							(225.0, 170.0, -135.0),      # Right eye right corne
							(-150.0, -150.0, -125.0),    # Left Mouth corner
							(150.0, -150.0, -125.0)      # Right mouth corner
						])
###############################################################################
###############################################################################

# Loop
while(True):
	# FRAME PROCESSING -----------------------------------------------

	# Imutil Video Read frame capture and manipulation
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)							#Dlib detector


	# Prints frame rate
	tl = time.time()
	t_diff = tl-t_temp
	cv2.putText(frame, "FPS {}".format(int(1/(t_diff))), (20,20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
	t_temp = tl

	# Face Detection -----------------------------------------------
	outsideBox = False
	personYawnRatio = False

	# detect faces in the grayscale image
	# for (x, y, w, h) in rects:
	for (j,rect)in enumerate(rects):
		i = 0
		(x, y, w, h) = rect_to_bb(rect)

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
		for (a, b) in shape:
			cv2.circle(frame, (a, b), 1, (0, 0, 255), -1)

		## YawnRatio or not?
		personYawnRatio = YawnRatio(shape) > YAWN_THRESH

		# Projection of points for gaze detection
		image_points, nose_end_point2D = project2Dto3D(shape,camera_matrix, model_points)

		## Gaze outside of bounding box or not?
		outsideBox = not inBoundingBox((nose_end_point2D[0][0][0], nose_end_point2D[0][0][1]),(x, y),(x + w, y + h), BOUNDING_BOX_THRESH)

		# Plots point on video montior
		p1 = ( int(image_points[0][0]), int(image_points[0][1]))
		p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		cv2.line(frame, p1, p2, (255,0,0), 2)

	# State processing + SerialToggle + SoundToggle ----------------------------

	# If face is found
	if len(rects) > 0:
		# For checking wakefullness
		eye1 = shape[lStart:lEnd]
		eye2 = shape[rStart:rEnd]
		# print("face")
		er1 = EyesRatio(eye1)
		er2 = EyesRatio(eye2)
		if (er1 < EAR_THRESH) and (er2 < EAR_THRESH):
			# Checks eyes for drowsiness
			state_temp = 'tired'
		elif outsideBox:
			state_temp = 'looking away'
		elif personYawnRatio:
			state_temp = 'YawnRatio'
		else:
			state_temp = 'clear'

	# If no face, that means distracted
	else:
		state_temp = 'distract'

	# Update state based on predetermined freq
	if (tl - t_state) > change_time:

		# If the state changes then act on periperals
		if state != state_temp:
			t_state = time.time()				# Record state change time

			# Sets distraction data for push to Database
			distraction_type = 0
			if state == 'clear':
				distraction_type = 0
				# Logs data to server args['user']
				if WebToggle: cli.log(1, distraction_type, datetime.datetime.now())
			elif state == 'distract':
				distraction_type = 1
				# Logs data to server args['user']
				if WebToggle: cli.log(1, distraction_type, datetime.datetime.now())
			# else:
			# 	distraction_type = 2

			if state_temp == 'looking away':
				print("looking away")
			if state_temp == 'YawnRatio':
				print("YawnRatio")
			# Driver is focused
			if state_temp == 'clear':
				if SerialToggle:
					client.off()
				else:
					print('clear')

			# Driver not looking at camera
			elif state_temp == 'distract':
				if SerialToggle:
					client.red()
				else:
					print('distact')
				if SoundToggle: playSoundToggle('beep-01a.mp3',block=False)

			state = state_temp					# Store new state

		# only plays distact SoundToggle if the state repeated as eyes closed
		elif state == 'tired':
			if SerialToggle:
				client.pulse(Color(0,255,0),3,10)
			else:
				print('tired')
			if SoundToggle: playSoundToggle('beep-02a.mp3',block=False)
			# Logs data to server args['user']
			state = ''
			distraction_type=2
			if WebToggle: cli.log(1, distraction_type, datetime.datetime.now())

		# Sends current state and timestamp to database



    # Display Frame and exit functionality -------------------------------
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


print('Aborted, runtime:'), time.time()-t0, ' (s)'
if SerialToggle: client.close()
if WebToggle: cli.close()
# cap.release()
cv2.destroyAllWindows()
vs.stop()
