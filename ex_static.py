#!/usr/bin/env python

import cv2
import numpy as np
import dlib
import imutils
from defs import *

'''
TO DO:
- There are currrently 3 modules, take photo, gaze estimation, and then face detections
- want to reorder so that gaze estimation works for any detected face
- then we make it realtime
- then we 'just' put it on the pico pro ha ha
- woo
'''


################################################################
# IMAGE CAPTURE ############################################################
# This is somethign I found online to open the mac webcam and take, save a photo
################################################################
cap = cv2.VideoCapture(0)

# Gets photo
while(True):
    ret, frame = cap.read()

    # May want to chane to grayscale eventually
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out = cv2.imwrite('capture.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()

################################################################
# Gaze Estimation ##########################################################
################################################################

# Read Image
im = cv2.imread("capture.jpg");
size = im.shape         # Functions later use this to calibrate camera

#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (658, 387),     # Nose tip
                            (660, 593),     # Chin
                            (537, 318),     # Left eye left corner
                            (774, 313),     # Right eye right corne
                            (591, 493),     # Left Mouth corner
                            (716, 492)      # Right mouth corner
                        ], dtype="double")
'''
Default
image_points = np.array([
                            (359, 391),     # Nose tip
                            (399, 561),     # Chin
                            (337, 297),     # Left eye left corner
                            (513, 301),     # Right eye right corne
                            (345, 465),     # Left Mouth corner
                            (453, 469)      # Right mouth corner
                        ], dtype="double")
'''

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])


# Camera internals
# NEED TO UPDATE FOR macpro
'''
Actual photo calibration is more involved.
Some people in one of the labs I am working on camera calibration tools (in python)
So, maybe we can take those eventually. The hardcoded values work okay, or maybe can find
 parameters online
'''
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

print "Camera Matrix :\n {0}".format(camera_matrix)

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

# SolvePnP is a known algorithm for extraction depth in 2D images
# https://en.wikipedia.org/wiki/Perspective-n-Point
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print "Rotation Vector:\n {0}".format(rotation_vector)
print "Translation Vector:\n {0}".format(translation_vector)


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

# For frawing the blue line in the example
(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(im, p1, p2, (255,0,0), 2)

# Display image
# cv2.imshow("Output", im)
# cv2.waitKey(0)

################################################################
# DLIB Example ################################################################
################################################################

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor, this is in the git repo, needed to download it
shape_predict = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predict)


# load the input image, resize it, and convert it to grayscale
# Can change image below
image = cv2.imread("capture.jpg");
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

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
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
print shape
