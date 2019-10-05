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
                            matches, flags=cv2.DrawMatchesFlags_DEFAULT,
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
    img_match_kp = cv2.drawMatches(img_left, kp_left, img_right, kp_right, matches, 
                None, flags=flags) 

    cv2.imshow(imshow_prefix + 'Matching keypoints', img_match_kp)
    cv2.waitKey(0) & 0xff

    cv2.destroyAllWindows()


# -----------------------------------
# Brute Force Matcher with ORB
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

draw_visualise_matcher(img_left, kp_left, img_right, kp_right, 
                        matches, flags=cv2.DrawMatchesFlags_DEFAULT,
                        imshow_prefix="BFMatcher+ORB: ")


# -----------------------------------
# Brute Force Matcher with ORB
# -----------------------------------

# sift is deprecated from opencv 3.4.2 onwards
sift = cv2.xfeatures2d.SIFT_create() # cv2.SIFT() for opencv 3.4.1 <

# find the keypoints and descriptors with SIFT
kp_left, desc_left = sift.detectAndCompute(img_left, None)
kp_right, desc_right = sift.detectAndCompute(img_right, None)

matches = bfmatcher.match(desc_left, desc_right)
matches = sorted(matches, key = lambda x:x.distance)

draw_visualise_matcher(img_left, kp_left, img_right, kp_right, 
                        matches, flags=cv2.DrawMatchesFlags_DEFAULT,
                        imshow_prefix="BFMatcher+SIFT: ")

# -----------------------------------
# Flann Based Matcher
# -----------------------------------




