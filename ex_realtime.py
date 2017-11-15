#!/usr/bin/env python

import cv2
import numpy as np
import dlib
import imutils
from defs import *
import time
from scipy.spatial import distance as dist
# Packages for serial
# import serial
# import sys
# from serial.tools.list_ports import comports as list_comports
from playsound import playsound

'''
TO DO:
- There are currrently 3 modules, take photo, gaze estimation, and then face detections
- want to reorder so that gaze estimation works for any detected face
- then we make it realtime
- then we 'just' put it on the pico pro ha ha
- woo
'''


'''
# Serial setup

port = None
for s in list_comports():
    print(s)
    if 'usbmodem' in s[0]:
        port = s[0]
        break

if port is None:
    print("NO SERIAL PORT COULD BE FOUND")
    sys.exit()
print("Using port " + port)
'''
################################################################
# Gaze Estimation ##########################################################
################################################################


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

    '''
    # Camera internals
    # NEED TO UPDATE FOR macpro

    Actual photo calibration is more involved.
    Some people in one of the labs I am working on camera calibration tools (in python)
    So, maybe we can take those eventually. The hardcoded values work okay, or maybe can find
    parameters online
    '''
    #2D image points. If you change the image, you need to change vector
    # image_points = np.array([
    #                             (658, 387),     # Nose tip
    #                             (660, 593),     # Chin
    #                             (537, 318),     # Left eye left corner
    #                             (774, 313),     # Right eye right corne
    #                             (591, 493),     # Left Mouth corner
    #                             (716, 492)      # Right mouth corner
    #                         ], dtype="double")
    #
    # # 3D model points.
    # model_points = np.array([
    #                             (0.0, 0.0, 0.0),             # Nose tip
    #                             (0.0, -330.0, -65.0),        # Chin
    #                             (-225.0, 170.0, -135.0),     # Left eye left corner
    #                             (225.0, 170.0, -135.0),      # Right eye right corne
    #                             (-150.0, -150.0, -125.0),    # Left Mouth corner
    #                             (150.0, -150.0, -125.0)      # Right mouth corner
    #
    #                         ])
    #
    # focal_length = size[1]
    # center = (size[1]/2, size[0]/2)
    # camera_matrix = np.array(
    #                         [[focal_length, 0, center[0]],
    #                         [0, focal_length, center[1]],
    #                         [0, 0, 1]], dtype = "double"
    #                         )

    # print "Camera Matrix :\n {0}".format(camera_matrix)
    '''
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    # SolvePnP is a known algorithm for extraction depth in 2D images
    # https://en.wikipedia.org/wiki/Perspective-n-Point
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print "Rotation Vector:\n {0}".format(rotation_vector)
    # print "Translation Vector:\n {0}".format(translation_vector)


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    # For frawing the blue line in the example
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    ## Points on nose
    print(nose_end_point2D)
    print(image_points)
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255,0,0), 2)
    '''
    # Display image
    # cv2.imshow("Output", im)
    # cv2.waitKey(0)

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



################################################################
# IMAGE CAPTURE ############################################################
# This is somethign I found online to open the mac webcam and take, save a photo
################################################################
cap = cv2.VideoCapture(0)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor, this is in the git repo, needed to download it
shape_predict = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predict)
# Gets photo

# Comment out serial such for computer only example
# with serial.Serial(port, 9200) as ser:
ear_thresh = .3
try:
    playflag = False
    while(True):
        ret, frame = cap.read()
        #analyzeImage(frame)
        # May want to chane to grayscale eventually
        image, faces, shape = analyzeImage(frame)
        eye1 = shape[36:42]
        eye2 = shape[42:48]
        ## if no faces in frame
        if len(faces) > 0:
            print("face")
            # ser.write('set 125 65280\n')
            playflag = True
            # ser.write('set 125 4370175\n')
            ear1 = eye_aspect_ratio(eye1)
            ear2 = eye_aspect_ratio(eye2)
            if (ear1 < ear_thresh) & (ear2 < ear_thresh):
                print("wake Up!")

        else:
            print("no face")
            # ser.write('p 1671680 1 10\n')
            playflag # and playsound('beep-01a.mp3', block=False)
            playflag = False

            ## Change color
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        ## No Face sh
        # time.sleep(0.1)
        cv2.imshow('frame', rgb)

        # Communicates

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
    # ser.write("off\n")
    cap.release()
    cv2.destroyAllWindows()
