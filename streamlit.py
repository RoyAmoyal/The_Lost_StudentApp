import cv2 as cv
import numpy as np
import streamlit as st

def sift_feature_matching(image1_path, image2_path):
    # Load images
    img1 = cv.imread(image1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image2_path, cv.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Perform feature matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    matching_result = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    st
    len(good_matches)
    st.image(cv.resize(matching_result, (1280, 720), cv.INTER_AREA))

# Example usage
image1_path = "images/97/20240416_105151.jpg"
image2_path = "images/lib_22/20240416_104927.jpg"
image3_path = "images/lib_22/20240416_104928.jpg"

sift_feature_matching(image1_path, image2_path)
sift_feature_matching(image2_path, image3_path)