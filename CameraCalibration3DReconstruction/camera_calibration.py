""" CAMERA CALIBRATION TUTORIAL

The Goal for this tutorial will be to help you learn about camera distortions 
that are typically present in photos taken with common pinhole cameras. 
We will also learn the definition and differences between intrinsic vs extrinsic 
parameters of the camera and why they are needed in our code. 
Once these parameters are found, we can use Open CV to undistort the image. 
This is the first step towards full 3D reconstruction.
"""

import glob

import cv2
import numpy as np


# -----------------------------------
# CAMERA CALIBRATION TUTORIAL
# -----------------------------------

# termination criteria for refining corners pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# initialise grid object points of shape (7 x 6)
pshape = (7 * 6, 3)
objp = np.zeros(pshape, np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# array to store object points and image points for all images 
objpoints_list = []  # 3d point in real space 
imgpoints_list = []  # 2d point in image plane

# load image of checker boards
imagepaths = glob.glob("data/left*[0-9].jpg")

# find chessboard corners for every image in folders './data'
for fname in imagepaths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find corner of checker/chess board
    success, corners = cv2.findChessboardCorners(img, (7,6), cv2.CALIB_CB_ADAPTIVE_THRESH)

    if success == True:
        objpoints_list.append(objp)
        
        # refining points of found corners
        corners2  = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints_list.append(corners)

        # draw chessboard corners on image for visualisation
        cv2.drawChessboardCorners(img, (7,6), corners2, success)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

img = cv2.imread('data/left11.jpg') # pick an image to demo calibration
height, width, channel = img.shape

# camera calibration using image points and object points
# yield: camera matrix, distortion coef, rotation & translation vectors
success, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, 
    imgpoints_list, (width, height), None, None)

# undistortion
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, 
    (width,height), 1, (width,height))
dist = cv2.undistort(img, matrix, distortion, None, newcameramtx)

# crop the image
x, y, w, h = roi
dist = dist[y:y+h, x:x+w]
cv2.imwrite('artifacts/calibresult11.png', dist)

cv2.imshow('calibresult', dist)
cv2.waitKey(500)

# save matrix, distortion coef, rotation, translation
# with np.savez()
outfile = "artifacts/calibresult.npz"
np.savez(outfile, mtx=matrix, distr=distortion, rvecs=rvecs, tvecs=tvecs)
