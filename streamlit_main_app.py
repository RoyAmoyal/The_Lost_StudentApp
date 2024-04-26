import concurrent
import concurrent.futures

import pickle

import cv2 as cv
import cv2
import kornia
import numpy as np
import os
import streamlit as st
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *
from collections import defaultdict

import cv2
from io import BytesIO
device = K.utils.get_cuda_or_mps_device_if_available()

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return KF.DISK.from_pretrained("depth").to(device), KF.LightGlueMatcher("disk").eval().to(device),

# lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)
#
# disk = KF.DISK.from_pretrained("depth").to(device)
disk,lg_matcher = load_model()
num_features = 2048
def white_balance_grayworld(image):
    avg_b = np.mean(image[:,:,0])
    avg_g = np.mean(image[:,:,1])
    avg_r = np.mean(image[:,:,2])

    avg_gray = np.mean([avg_b, avg_g, avg_r])

    # Calculate scaling factors for each channel
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # Apply scaling factors to each channel
    balanced_image = np.zeros_like(image, dtype=np.float32)
    balanced_image[:,:,0] = np.clip(image[:,:,0] * scale_b, 0, 255)
    balanced_image[:,:,1] = np.clip(image[:,:,1] * scale_g, 0, 255)
    balanced_image[:,:,2] = np.clip(image[:,:,2] * scale_r, 0, 255)

    return balanced_image.astype(np.uint8)
# Function to save keypoints and descriptors to file
@st.cache_data
def save_keypoints_descriptors_to_file(keypoints_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(keypoints_dict, f)

# Function to load keypoints and descriptors from file
@st.cache_data
def load_keypoints_descriptors_from_file(file_path):
    with open(file_path, 'rb') as f:
        keypoints_dict = pickle.load(f)
    return keypoints_dict
def extract_keypoints_and_descriptors(img):
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    with torch.inference_mode():
        # inp = torch.cat([img1, img2], dim=0)
        features1 = disk(img, num_features, pad_if_not_divisible=True)[0]
        kps1, descs1 = features1.keypoints, features1.descriptors
        # kps2, descs2 = features2.keypoints, features2.descriptors
        # lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))

    return kps1, descs1
def process_image(image_path):
    image = cv.imread(image_path)
    image = white_balance_grayworld(image)
    image = kornia.image_to_tensor(image, keepdim=False)
    image = kornia.color.bgr_to_rgb(image)
    image = K.geometry.resize(image, (640, 480)) / 255
    keypoints, descriptors = extract_keypoints_and_descriptors(image)
    return keypoints, descriptors
def extract_keypoints_descriptors_dict(images_folder):
    keypoints_dict = {}
    progress_text = "Doing some heavy lifting know so you won't have to wait later."
    progress_bar = st.progress(0, text=progress_text)
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    current_image_count = 0

    image_paths = []
    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        for image_name in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, image_name))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_image, image_paths)

    idx = 0
    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        folder_data = {}

        for image_name in os.listdir(folder_path):
            keypoints, descriptors = next(results)
            folder_data[image_name] = {'keypoints': keypoints, 'descriptors': descriptors}
            idx += 1
            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)

        keypoints_dict[folder_name] = folder_data

    progress_bar.empty()
    return keypoints_dict


# @st.cache_resource(show_spinner = False)
# def extract_keypoints_descriptors_dict(images_folder):
#     keypoints_dict = {}
#     progress_text = "Doing some heavy lifting know so you won't have to wait later."
#     progress_bar = st.progress(0, text=progress_text)
#     total_images = sum(len(files) for _, _, files in os.walk(images_folder))
#     current_image_count = 0
#
#     for folder_name in os.listdir(images_folder):
#         folder_path = os.path.join(images_folder, folder_name)
#
#         folder_data = {}
#
#         for image_name in os.listdir(folder_path):
#             image_path = os.path.join(folder_path, image_name)
#             image = cv.imread(image_path)
#             image = white_balance_grayworld(image)
#
#             # image = white_balance_grayworld(image)
#
#             # image = cv.resize(image, (1280,720),cv.INTER_LINEAR)
#             image = kornia.image_to_tensor(image, keepdim=False)
#
#             image = kornia.color.bgr_to_rgb(image)
#             image = K.geometry.resize(image, (640, 480)) / 255
#             keypoints, descriptors = extract_keypoints_and_descriptors(image)
#             folder_data[image_name] = {'keypoints': keypoints, 'descriptors': descriptors}
#
#             current_image_count += 1
#             progress_bar.progress(current_image_count / total_images, text=progress_text)
#
#         keypoints_dict[folder_name] = folder_data
#     progress_bar.empty()
#     return keypoints_dict
def match_keypoints(descriptors1, descriptors2):
    bf = cv.BFMatcher()

    # bf = cv.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))
    bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    return good_matches.copy()
#
# def find_top_matches(input_keypoints,input_descriptors, keypoints_dict, images_folder, top_x=20):
#     def get_matching_keypoints(kp1, kp2, idxs):
#         mkpts1 = kp1[idxs[:, 0]]
#         mkpts2 = kp2[idxs[:, 1]]
#         return mkpts1, mkpts2
#     top_matches = []
#     progress_text = "Finding where you are, hold on tight!"
#     progress_bar = st.progress(0, text=progress_text)
#     total_images = sum(len(files) for _, _, files in os.walk(images_folder))
#     current_image_count = 0
#
#     for folder_name, folder_data in keypoints_dict.items():
#         for image_name, image_data in folder_data.items():
#             image_path = os.path.join(images_folder, folder_name, image_name)
#             # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#             # inp = torch.cat([img1, img2], dim=0)
#             kps1, descs1 = image_data['keypoints'],  image_data['descriptors']
#
#             dists, idxs = lg_matcher(descs1, input_descriptors, KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device)), KF.laf_from_center_scale_ori(input_keypoints[None], torch.ones(1, len(input_keypoints), 1, 1, device=device)))
#
#             mkpts1, mkpts2 = get_matching_keypoints(kps1, input_keypoints, idxs)
#
#             Fm, inliers = cv2.findFundamentalMat(
#                 mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.999, 100000
#             )
#             inliers = inliers > 0
#             # inliers1,inliers2,good_matches = find_inliers(input_keypoints,input_descriptors,image_data['keypoints'],image_data['descriptors'])
#             # good_matches = match_keypoints(input_descriptors.copy(), descriptors.copy())
#             # print(inliers.shape)
#             match_count = inliers.shape[0]
#             # l2_distance = sum([m.distance for m in good_matches]) if good_matches else float('inf')
#             top_matches.append((folder_name, image_name, match_count, 0, inliers,kps1,descs1,idxs))
#
#             current_image_count += 1
#             progress_bar.progress(current_image_count / total_images, text=progress_text)
#     progress_bar.empty()
#
#     top_matches = sorted(top_matches,key=lambda x: x[2], reverse=True)
#     top_matches = top_matches[:top_x] if top_x > 0 else top_matches
#
#     return top_matches
import os
import cv2
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2
def process_image(folder_name, image_name, image_data, input_keypoints, input_descriptors):
    kps1, descs1 = image_data['keypoints'],  image_data['descriptors']

    # Your processing logic here
    # For simplicity, let's just calculate the match count as the number of keypoints
    # hw1 = torch.tensor((640,480), device=device)
    # hw2 = torch.tensor((640,480), device=device)

    dists, idxs = lg_matcher(descs1, input_descriptors, KF.laf_from_center_scale_ori(kps1[None],
                                                                                     torch.ones(1, len(kps1), 1, 1,
                                                                                                device=device)),
                             KF.laf_from_center_scale_ori(input_keypoints[None],
                                                          torch.ones(1, len(input_keypoints), 1, 1, device=device)))

    mkpts1, mkpts2 = get_matching_keypoints(kps1, input_keypoints, idxs)

    Fm, inliers = cv2.findFundamentalMat(
        mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    inliers = inliers > 0

    match_count = inliers.shape[0]

    return (folder_name, image_name, match_count, 0, inliers, kps1, descs1, idxs)
def find_top_matches(input_keypoints, input_descriptors, keypoints_dict, images_folder, top_x=5):
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    progress_text = "Finding where you are, hold on tight!"
    current_image_count = 0
    top_matches = []

    def update_progress(result):
        nonlocal current_image_count
        current_image_count += 1
        print(f"{progress_text} ({current_image_count}/{total_images})")
        top_matches.append(result)

    with Pool(cpu_count()) as pool:
        for folder_name, folder_data in keypoints_dict.items():
            for image_name, image_data in folder_data.items():
                pool.apply_async(process_image, args=(folder_name, image_name, image_data, input_keypoints, input_descriptors), callback=update_progress)
        pool.close()
        pool.join()

    top_matches.sort(key=lambda x: x[2], reverse=True)
    top_matches = top_matches[:top_x] if top_x > 0 else top_matches

    return top_matches
@st.cache_data
def load_image_from_web(uploaded_file):
    input_image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
    return input_image

def main():
    st.title('The Lost Student ICVL Project')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace("\\", "/")
    print(dir_path)
    images_folder = dir_path + "/The_Lost_Student_App/images"
    images_folder = "D:/University/msc/semester a/intro to computational biology and vision/final_project/The_Lost_Student_App/images"

    keypoints_file = "keypoints_descriptors.pkl"  # File to save/load keypoints and descriptors

    # Check if keypoints and descriptors file exists, load them if it does
    if os.path.exists(keypoints_file):
        keypoints_dict = load_keypoints_descriptors_from_file(keypoints_file)
    else:
        keypoints_dict = extract_keypoints_descriptors_dict(images_folder)
        # Save keypoints and descriptors to file
        save_keypoints_descriptors_to_file(keypoints_dict, keypoints_file)

    # uploaded_file = True
    if uploaded_file is not None:
        input_image = load_image_from_web(uploaded_file)
        st.image(input_image[:, :, ::-1], caption='Uploaded Image', use_column_width=True)
        input_image = white_balance_grayworld(input_image)

        # input_image = cv.imread('images/97/20240416_105208.jpg')  # queryImage
        # input_image = cv.imread('../48f4f37a-d51b-46cf-93a9-1c3c6f737872.jpg')  # queryImage
        # input_image = cv.imread('what.jpg')  # queryImage

        # input_image = white_balance_grayworld(input_image)
        # st.image(input_image[:, :, ::-1], caption='Uploaded Image', use_column_width=True)

        std_size = (640,480)
        # input_image = cv.resize(input_image, std_size)
        # input_image = cv.resize(input_image, (1280,720), cv.INTER_LINEAR)
        # input_image = cv.cvtColor(input_image,cv.COLOR_BGR2GRAY)
        input_image = kornia.image_to_tensor(input_image,keepdim=False)
        input_image = kornia.color.bgr_to_rgb(input_image)
        input_image = K.geometry.resize(input_image, (640, 480)) / 255
        input_keypoints, input_descriptors = extract_keypoints_and_descriptors(input_image)
        top_matches = find_top_matches(input_keypoints,input_descriptors, keypoints_dict, images_folder)
        count_dict = defaultdict(int)
        for idx, (folder_name, image_name, match_count, _, top_match_keypoints,kps1,descs1,idxs) in enumerate(top_matches):
            count_dict[folder_name] += 1

            st.write(f"Match {idx+1}: {folder_name}/{image_name}")
            st.write(f"Number of matches: {match_count}")

            image_path = os.path.join(images_folder, folder_name, image_name)
            match_image = cv.imread(image_path)
            # match_image = white_balance_grayworld(match_image)
            # match_image = cv.resize(match_image, (1280,720),cv.INTER_LINEAR)
            match_image = kornia.image_to_tensor(match_image, keepdim=False)
            match_image = kornia.color.bgr_to_rgb(match_image)
            match_image = K.geometry.resize(match_image, (640, 480)) / 255
            match_keypoints = keypoints_dict[folder_name][image_name]['keypoints']
            print(top_match_keypoints.shape)
            print(idxs.shape)
            print(KF.laf_from_center_scale_ori(input_keypoints[None].cpu()).shape)
            print(KF.laf_from_center_scale_ori(match_keypoints[None].cpu()).shape)

            draw_LAF_matches(
                KF.laf_from_center_scale_ori(match_keypoints[None].cpu()),
                KF.laf_from_center_scale_ori(input_keypoints[None].cpu()),
                idxs.cpu(),
                K.tensor_to_image(match_image.cpu()),
                K.tensor_to_image(input_image.cpu()),
                top_match_keypoints,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": (1, 1, 0.2, 0.3), "feature_color": None,
                           "vertical": False}, return_fig_ax=True
            )

            # Assuming your matplotlib plot is generated as before

            # Render the matplotlib plot to a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Convert the buffer to a numpy array
            buffer_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            plt.close()  # Close the matplotlib plot to free resources

            # Convert the numpy array to an OpenCV image
            opencv_img = cv2.imdecode(buffer_img, 1)
            # img_matches = cv.drawMatches(input_image, input_keypoints, match_image, match_keypoints, top_match_keypoints, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(opencv_img[:, :, ::-1], caption=f'Match {idx+1}: {folder_name}/{image_name}', use_column_width=True)
        most_common_folder = max(count_dict, key=count_dict.get)
        print("THE FOLDER WINNER IS ",most_common_folder)
        st.write("THE FOLDER WINNER IS ", most_common_folder)
if __name__ == "__main__":
    main()