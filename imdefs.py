import numpy as np
from scipy.spatial import distance as dist
import cv2

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def inBoundingBox(nosePoint, rectangleBase, rectangleEnd, BOUNDING_BOX_ERROR):
	rectangleWidthError = abs(rectangleBase[0] - rectangleEnd[0]) *  BOUNDING_BOX_ERROR
	rectangleHeightError = abs(rectangleBase[1] - rectangleEnd[1]) * BOUNDING_BOX_ERROR

	greaterThanx = nosePoint[0] >= (rectangleBase[0] - rectangleWidthError)
	lessThanBoundX = nosePoint[0] <= (rectangleEnd[0] + rectangleWidthError)
	greaterThanY = nosePoint[1] >= (rectangleBase[1] - rectangleHeightError)
	lessThanBoundY = nosePoint[1] <= (rectangleEnd[1] + rectangleHeightError)

	return greaterThanx and lessThanBoundX and greaterThanY and lessThanBoundY


def EyesRatio(eye):
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

## yawning or not
def YawnRatio(shape):
	height = dist.euclidean(shape[52], shape[58])
	width = dist.euclidean(shape[55], shape[49])
	return height/width

def project2Dto3D(shape,camera_matrix,model_points):
	# Gaze detector + display
	'''
	Default
	image_points = np.array([
									# Nose tip
									# Chin
									# Left eye left corner
									# Right eye right corne
									# Left Mouth corner
									 # Right mouth corner
							], dtype="double")
	'''

	# Specific features to transform to 3d
	image_points = np.array([
							shape[33],
							shape[8],
							shape[36],
							shape[45],
							shape[48],
							shape[54]
							], dtype="double")


	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
	# For frawing the blue line in the example
	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 100.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

	return image_points, nose_end_point2D
