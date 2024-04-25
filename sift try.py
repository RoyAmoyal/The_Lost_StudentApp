import cv2
import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

img1 = cv.imread('images_check/4.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('images_check/5.jpg', cv.IMREAD_GRAYSCALE)  # trainImage
img1 = cv.imread('images/25/20240416_104922.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
if img1.shape[0]/img2.shape[0] > 2.2:
    img2 = cv2.resize(img2,(img1.shape[1]*2,img1.shape[0]*2),cv2.INTER_CUBIC)
elif img2.shape[0] / img1.shape[0] > 2.2:
    img1 = cv2.resize(img1, (img2.shape[1] * 2, img2.shape[0] * 2), cv2.INTER_CUBIC)
# img1 = cv2.resize(img1,(2000,1500),cv.INTER_LINEAR)
# img2 = cv2.resize(img2,(2000,1500),cv.INTER_LINEAR)

#
# img1 = cv.imread('images_check/20240416_105208.jpg.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('images_check/2.jpg', cv.IMREAD_GRAYSCALE)  # trainImage


# Initiate SIFT detector
sift = cv.SIFT_create()
bf = cv.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
# bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
print(len(good))
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.resize(img3,(1280,720),cv2.INTER_LINEAR)
cv2.imshow("w",img3)
cv2.waitKey(0)