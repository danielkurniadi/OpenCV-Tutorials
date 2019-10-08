""" EPIPOLAR GEOMETRY TUTORIAL

In this section: 
- We will learn about the basics of multiview geometry
- We will see what is epipole, epipolar lines, epipolar constraint etc.
"""
import os
import glob

import cv2
import numpy as np


# -----------------------------------
# EPIPOLAR GEOMETRY TUTORIAL
# -----------------------------------

img_left = cv2.imread('./data/left_real.jpg', cv2.IMREAD_GRAYSCALE) # query image
img_right = cv2.imread('./data/right_real.jpg', cv2.IMREAD_GRAYSCALE) # train image


# Before we begin ...
def draw_matching_keypoints(img_left, kp_left, img_right, kp_right, 
							matches, ratio_test=False, flags=cv2.DrawMatchesFlags_DEFAULT,
							imshow_prefix="Random shit:"):
	""" Draw matching keypoints from keypoints and descriptor of two images.
	Ideally the two images are photograph of the same object but taken at different pov

	:param img_left (np image): the first input image 
	:param kp_left (np array): keypoints of the first image
	:param img_right (np image): the second input image
	:param kp_right (np array): keypoints of the second image
	:param matches (np array): matching points from descriptor matching of first and second image

	Note: Draw Function Flags:
		- DEFAULT = 0
		- DRAW_OVER_OUTIMG = 1
		- NOT_DRAW_SINGLE_POINTS = 2
		- DRAW_RICH_KEYPOINTS = 4
	"""

	# draw keypoints on the left image
	img_left_kp = cv2.drawKeypoints(img_left, kp_left, None)

	cv2.imshow(imshow_prefix + 'Image Left keypoints', img_left_kp)
	cv2.waitKey(0) & 0xff

	# draw keypoints on the right image
	img_right_kp = cv2.drawKeypoints(img_right, kp_right, None)
	cv2.imshow(imshow_prefix + 'Image Right keypoints', img_right_kp)
	cv2.waitKey(0) & 0xff

	# draw matching keypoints
	if ratio_test:
		img_match_kp = cv2.drawMatchesKnn(img_left, kp_left, img_right, kp_right, matches, 
				None, flags=2)
	else:
		img_match_kp = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches, 
				None, flags=flags) 

	cv2.imshow(imshow_prefix + 'Matching keypoints', img_match_kp)
	cv2.waitKey(0) & 0xff

	cv2.destroyAllWindows()


# -----------------------------------
# Brute Force Matcher + ORB
# -----------------------------------

orb = cv2.ORB_create()

# combine detector and descriptor, hence cv2::detectAndComputer
# calculating: keypoints, descriptor
kp_left, desc_left = orb.detectAndCompute(img_left, None)
kp_right, desc_right = orb.detectAndCompute(img_right, None)

# create descriptor matcher for matching keypoints in left and right image
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# calculate match descriptor and sort it based on distance
# match_descr = bfmatcher.match(desc_left, desc_right)
matches = bfmatcher.match(desc_left, desc_right)
matches = sorted(matches, key = lambda x:x.distance)

draw_matching_keypoints(img_left, kp_left, img_right, kp_right, 
						matches[:10], flags=cv2.DrawMatchesFlags_DEFAULT,
						imshow_prefix="BFMatcher+ORB: ")


# -----------------------------------
# Brute Force Matcher (KNN) + SIFT
# -----------------------------------

# sift is deprecated from opencv 3.4.2 onwards
sift = cv2.xfeatures2d.SIFT_create() # cv2.SIFT() for opencv 3.4.1 <

# find the keypoints and descriptors with SIFT
kp_left, desc_left = sift.detectAndCompute(img_left, None)
kp_right, desc_right = sift.detectAndCompute(img_right, None)

bfmatcher = cv2.BFMatcher()
matches = bfmatcher.knnMatch(desc_left, desc_right, k=2)

good_matches = []
points_left = []
points_right = []
thresh_ratio = 0.65

# using ratio tests as in Lowe's paper
for i, (m,n) in enumerate(matches):
	if m.distance < thresh_ratio* n.distance:
		good_matches.append([m])
		points_right.append(kp_right[m.trainIdx].pt)
		points_left.append(kp_left[m.queryIdx].pt)

# draw matching keypoints
draw_matching_keypoints(img_left, kp_left, img_right, kp_right, 
						good_matches[:10], ratio_test=True, 
						flags=cv2.DrawMatchesFlags_DEFAULT,
						imshow_prefix="BFMatcher+SIFT: ")


# -----------------------------------
# Flann Based Matcher + SIFT
# -----------------------------------

# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm=FLANN_INDEX_LSH,
# 				   table_number=6, # 12
# 				   key_size=12,     # 20
# 				   multi_probe_level=1) #2

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp_left, desc_left = sift.detectAndCompute(img_left, None)
kp_right, desc_right = sift.detectAndCompute(img_right, None)

# FLANN parameters
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_left, desc_right, k=2)

good_matches = []
points_left = []
points_right = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.45 * n.distance:
        good_matches.append([m])
        points_right.append(kp_right[m.trainIdx].pt)
        points_left.append(kp_left[m.queryIdx].pt)

# now given best matches from both image,
# calculate the fundamental matrix and mask
points_left = np.int32(points_left)
points_right = np.int32(points_right)
F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_LMEDS)

# create inliers by using mask
inliers_left = points_left[mask.ravel() == 1]
inliers_right = points_right[mask.ravel() == 1]


# define function helper to draw epilines on first image corresponding to
# matching points in second image
def draw_epilines(img1, img2, epilines, points1, points2):
	"""
	:param img1 (np image): image which epilines is to be annotated.
	:param img2 (np image): image which points is found in the epilines of the 1st image
	:param epilines (np array): line on 1st image which contains projection points from 2nd image
	:param points1 (np array): matched keypoint on 1st image
	:param points2 (np array): matched keypoint on 2nd image
	"""
	h,w = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	for r, pt1, pt2 in zip(epilines, points1, points2):
		color = tuple(np.random.randint(0, 255, 3).tolist())
		
		x0,y0 = map(int, [0, -r[2] / r[1]])
		x1,y1 = map(int, [w, -(r[2] + r[0] *w) / r[1]])

		img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
		img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
		img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
	
	return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
epilines1 = cv2.computeCorrespondEpilines(points_right.reshape(-1,1,2), 2, F)
epilines1 = epilines1.reshape(-1,3)
img5, img6 = draw_epilines(img_left, img_right, epilines1, points_left, points_right)

epilines2 = cv2.computeCorrespondEpilines(points_left.reshape(-1,1,2), 1, F)
epilines2 = epilines2.reshape(-1,3)
img3, img4 = draw_epilines(img_right, img_left, epilines2, points_right, points_left)

cv2.imshow('Image left epiline', img5)
cv2.imshow('Image right epiline', img3)

cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()
