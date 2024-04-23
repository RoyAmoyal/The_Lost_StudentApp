import cv2 as cv
import numpy as np
import os
import streamlit as st
from stqdm import stqdm

def extract_keypoints_and_descriptors(img, resize_factor=0.25, interpolation=cv.INTER_AREA):
    # if resize_factor != 1.0:
        # img_resized = cv.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=interpolation)
    # else:
        # img_resized = img

    # orb = cv.ORB_create()
    # kp = orb.detect(img,None)
    # keypoints, descriptors = orb.compute(img, kp)

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # if resize_factor != 1.0:
    #     scale_x = img.shape[1] / img_resized.shape[1]
    #     scale_y = img.shape[0] / img_resized.shape[0]
    #     for kp in keypoints:
    #         kp.pt = (kp.pt[0] * scale_x, kp.pt[1] * scale_y)

    return keypoints, descriptors

@st.cache_resource(show_spinner = False)
def extract_keypoints_descriptors_dict(images_folder):
    keypoints_dict = {}
    progress_text = "Doing some heavy lifting know so you won't have to wait later."
    progress_bar = st.progress(0, text=progress_text)
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    current_image_count = 0

    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        
        folder_data = {}
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv.imread(image_path)
            keypoints, descriptors = extract_keypoints_and_descriptors(image)
            folder_data[image_name] = {'keypoints': keypoints, 'descriptors': descriptors}

            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)

        keypoints_dict[folder_name] = folder_data
    progress_bar.empty()
    return keypoints_dict

def match_keypoints(descriptors1, descriptors2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_top_matches(input_descriptors, keypoints_dict, images_folder, top_x=10):
    top_matches = []
    progress_text = "Finding where you are, hold on tight!"
    progress_bar = st.progress(0, text=progress_text)
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    current_image_count = 0

    for folder_name, folder_data in keypoints_dict.items():
        for image_name, image_data in folder_data.items():
            descriptors = image_data['descriptors']
            good_matches = match_keypoints(input_descriptors, descriptors)
            match_count = len(good_matches)
            l2_distance = sum([m.distance for m in good_matches]) if good_matches else float('inf')
            top_matches.append((folder_name, image_name, match_count, l2_distance, good_matches))
            
            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)
    progress_bar.empty()

    top_matches.sort(key=lambda x: (x[2], x[3]), reverse=True)
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
        
        std_size = (4032, 1816)
        input_image = cv.resize(input_image, std_size)
        input_keypoints, input_descriptors = extract_keypoints_and_descriptors(input_image)
        top_matches = find_top_matches(input_descriptors, keypoints_dict, images_folder)
        
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