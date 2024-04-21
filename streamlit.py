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

def find_top_matches(input_descriptors, keypoints_dict, top_x=-1):
    top_matches = []
    
    for folder_name, folder_data in keypoints_dict.items():
        for image_name, image_data in folder_data.items():
            descriptors = image_data['descriptors']
            good_matches = match_keypoints(input_descriptors, descriptors)
            match_count = len(good_matches)
            l2_distance = sum([m.distance for m in good_matches]) if good_matches else float('inf')
            top_matches.append((folder_name, image_name, match_count, l2_distance, good_matches))
    
    top_matches.sort(key=lambda x: (x[2], x[3]))
    top_matches = top_matches[:top_x] if top_x > 0 else top_matches
    
    return top_matches

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
            top_matches = find_top_matches(input_descriptors, keypoints_dict)
        
        for idx, (folder_name, image_name, match_count, _, top_match_keypoints) in enumerate(top_matches):
            st.write(f"Match {idx+1}: {folder_name}/{image_name}")
            st.write(f"Number of matches: {match_count}")

            image_path = os.path.join(images_folder, folder_name, image_name)
            match_image = cv.imread(image_path)
            match_keypoints, _ = extract_keypoints_and_descriptors(match_image)

            img_matches = cv.drawMatches(input_image, input_keypoints, match_image, match_keypoints, top_match_keypoints, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(img_matches[:, :, ::-1], caption=f'Match {idx+1}: {folder_name}/{image_name}', use_column_width=True)

if __name__ == "__main__":
    main()