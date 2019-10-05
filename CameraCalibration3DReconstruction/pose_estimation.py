""" POSE ESTIMATION TUTORIAL

The Goal for this tutorial is to help understand how to use calib3d module to create some 
3D effects in image. Given a pattern image, we can utilize the above information to calculate its pose, or how the object is situated in space, like how it is rotated, how it is displaced etc. For a planar object, we can assume Z=0, such that, 
the problem now becomes how camera is placed in space to see our pattern image

Note: Not to be confused with pose estimation in OpenPose 
(https://github.com/CMU-Perpetual-Computing-Labs/openpose)
"""
import os
import glob

import cv2
import numpy as np


# -----------------------------------
# 3D POSE ESTIMATION TUTORIAL
# -----------------------------------

# load previous result from camera calibration
# you're recommended to follow camera_calibration tutorial first
with np.load("artifacts/calibresult.npz") as X:
	matrix, distortion, rvecs, tvecs = (X[var] for var in ("mtx", "distr", "rvecs", "tvecs"))


# define functions to draw 3D axis from given checkboard corners and axis points
def draw_3D_axis(img, origin, axispoints):
	"""Draw 3D axis from given checkboard corners and axis points
	:param img (np array): image to be anotated with 3D drawing
	:param origin (np array): x,y position of the first corner
	:param projpoints (np array): projected 3D points to image plane
	"""
	img = cv2.line(img, tuple(origin), tuple(axispoints[0].ravel()), (255,0,0), 5) # draw z axis line
	img = cv2.line(img, tuple(origin), tuple(axispoints[1].ravel()), (0,255,0), 5) # draw x axis line
	img = cv2.line(img, tuple(origin), tuple(axispoints[2].ravel()), (0,0,255), 5) # draw y axis line

	return img


def draw_3D_cube(img, origin, axispoints):
	"""Draw 3D cube from given checkboard corners and axis points
	:param img (np array): image to be anotated with 3D drawing
	:param origin (np array): x,y position of the first corner
	:param projpoints (np array): projected 3D points to image plane
	"""
	# axis from projpoints:
	# [[0,0,0], [0,3,0], [3,3,0], [3,0,0],
	#  [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]]
	axispoints = np.int32(axispoints).reshape(-1,2)

	# draw ground floor in green
	img = cv2.drawContours(img, [axispoints[:4]], -1, (0,255,0), -3)

	# draw pillars in blue color
	for i,j in zip(range(4),range(4,8)):
		img = cv2.line(img, tuple(axispoints[i]), tuple(axispoints[j]), (255), 3)

	# draw top layer in red color
	img = cv2.drawContours(img, [axispoints[4:]], -1, (0,0,255), 3)

	return img


# termination criteria for corner subpixel 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# object point 
objp = np.zeros((7*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# we will draw axis of length 3 where units will be in terms of chess square size 
# since we calibrated based on that size. 
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisCube = np.float32(
	[[0,0,0], [0,3,0], [3,3,0], [3,0,0],
	[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

imagepaths = glob.glob("data/*.jpg")

# find chessboard corners for every checkerboard in folders './data'
for fname in imagepaths:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	success, corners = cv2.findChessboardCorners(img, (7,6), cv2.CALIB_CB_ADAPTIVE_THRESH)

	if success == True:
		corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

		# Find the rotation and translation vectors.
		_, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, matrix, distortion)

		# project 3D points to image plane
		axispoints, jac = cv2.projectPoints(axis, rvecs, tvecs, matrix, distortion)

		# project 3D points to 8 corners in 3D space
		cubepoints, jac = cv2.projectPoints(axisCube, rvecs, tvecs, matrix, distortion)

		# draw axis of 3D projection in image
		origin = corners2[0].ravel()
		img1 = draw_3D_axis(img, origin, axispoints)
		cv2.imshow('3D axis: ' + fname, img1)

		img2 = draw_3D_cube(img, origin, cubepoints)
		cv2.imshow('3D cube axis: ' + fname, img2)
		k = cv2.waitKey(0) & 0xff

		if k == 's':
			fname, ext = os.path.splitext(fname)
			cv2.imwrite('artifacts/' + fname + '.png', img)

	else:
		print("Warning! Image not successful %s" %fname)

cv2.destroyAllWindows()

