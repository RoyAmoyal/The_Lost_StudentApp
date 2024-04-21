import cv2 as cv
import numpy as np
import os
import streamlit as st

def extract_keypoints_and_descriptors_path(image_path, res=(640, 480)):
    img = cv.imread(image_path)
    img_resized = cv.resize(img, res)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_resized, None)
    return keypoints, descriptors

def extract_keypoints_and_descriptors_img(img, res=(640, 480)):
    img_resized = cv.resize(img, res)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_resized, None)
    return keypoints, descriptors


def extract_keypoints_descriptors_dict(images_folder):
    keypoints_dict = {}

    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        
        folder_data = {}
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            keypoints, descriptors = extract_keypoints_and_descriptors_path(image_path)
            folder_data[image_name] = {'keypoints': keypoints, 'descriptors': descriptors}
        
        keypoints_dict[folder_name] = folder_data

    return keypoints_dict

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
    len(good_matches)
    st.image(cv.resize(matching_result, (1280, 720), cv.INTER_AREA))

def main(keypoints_dict):
    st.title('The Lost Student ICVL Project')    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
        keypoints, descriptors = extract_keypoints_and_descriptors_img(image)
        st.image(image)

if __name__ == "__main__":
    images_folder = "/home/omrihir/The_Lost_Student_App/images"
    keypoints_dict = extract_keypoints_descriptors_dict(images_folder)
    main(keypoints_dict)