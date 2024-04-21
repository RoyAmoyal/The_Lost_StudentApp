import cv2 as cv
import numpy as np
import os
import streamlit as st
from stqdm import stqdm

def extract_keypoints_and_descriptors(img, res=(1280, 960), interpolation=cv.INTER_AREA):
    img_resized = img
    # img_resized = cv.resize(img, res, interpolation=interpolation)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_resized, None)

    # for kp in keypoints:
        # kp.pt = (kp.pt[0] * img.shape[1] / res[0], kp.pt[1] * img.shape[0] / res[1])

    # scale_x = img.shape[1] / res[0]
    # scale_y = img.shape[0] / res[1]
    # if descriptors is not None:
        # for i in range(len(descriptors)):
            # descriptors[i, 0::4] *= scale_x
            # descriptors[i, 1::4] *= scale_y

    return keypoints, descriptors

@st.cache_resource
def extract_keypoints_descriptors_dict(images_folder):
    keypoints_dict = {}

    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        
        folder_data = {}
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv.imread(image_path)
            keypoints, descriptors = extract_keypoints_and_descriptors(image)
            folder_data[image_name] = {'keypoints': keypoints, 'descriptors': descriptors}
        
        keypoints_dict[folder_name] = folder_data

    return keypoints_dict

def match_keypoints(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_best_match(input_descriptors, keypoints_dict):
    best_match = None
    best_match_count = 0
    best_match_keypoints = []
    
    for folder_name, folder_data in keypoints_dict.items():
        for image_name, image_data in folder_data.items():
            descriptors = image_data['descriptors']
            good_matches = match_keypoints(input_descriptors, descriptors)
            match_count = len(good_matches)
            if match_count > best_match_count:
                best_match = (folder_name, image_name)
                best_match_count = match_count
                best_match_keypoints = good_matches
        
    return best_match, best_match_count, best_match_keypoints

def main():
    st.title('The Lost Student ICVL Project')    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    images_folder = "/home/omrihir/The_Lost_Student_App/images"
    keypoints_dict = extract_keypoints_descriptors_dict(images_folder)

    if uploaded_file is not None:
        input_image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
        st.image(input_image[:, :, ::-1], caption='Uploaded Image', use_column_width=True)
        
        input_keypoints, input_descriptors = extract_keypoints_and_descriptors(input_image)
        with st.spinner('Finding Where You Are...'):
            best_match, match_count, top_matches = find_best_match(input_descriptors, keypoints_dict)
        
        st.write(f"Best match: {best_match[0]}/{best_match[1]}")
        st.write(f"Number of matches: {match_count}")

        best_match_folder, best_match_image = best_match
        best_match_image_path = os.path.join(images_folder, best_match_folder, best_match_image)
        best_match_img = cv.imread(best_match_image_path)
        best_match_keypoints, _ = extract_keypoints_and_descriptors(best_match_img)

        img_matches = cv.drawMatches(input_image, input_keypoints, best_match_img, best_match_keypoints, top_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        st.image(img_matches[:, :, ::-1], caption=f'Best Match: {best_match[0]}/{best_match[1]}', use_column_width=True)

if __name__ == "__main__":
    main()