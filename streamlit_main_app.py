import concurrent
import concurrent.futures

import pickle
from threading import Lock

import cv2
import kornia
import streamlit as st
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *
from collections import defaultdict
import os
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from io import BytesIO
from stqdm import stqdm
import torchvision.transforms as T

lock = Lock()

def calculate_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # Normalize the histogram
    cv2.normalize(hist, hist)

    return hist.flatten()


def compare_histograms(hist1, hist2, threshold=0.15):
    # Compute the correlation coefficient between the histograms
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(correlation)
    # Check if the correlation coefficient is above the threshold
    if correlation > threshold:
        return True
    else:
        return False


@st.cache_resource
def load_model(device='cpu'):
    return KF.DISK.from_pretrained("depth").to(device), KF.LightGlueMatcher("disk").eval().to(device),

def white_balance_grayworld(image):
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    avg_gray = np.mean([avg_b, avg_g, avg_r])

    # Calculate scaling factors for each channel
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # Apply scaling factors to each channel
    balanced_image = np.zeros_like(image, dtype=np.float32)
    balanced_image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    balanced_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    balanced_image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)

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


def extract_keypoints_and_descriptors(img, disk, num_features=2048):
    with torch.inference_mode():
        features1 = disk(img, num_features, pad_if_not_divisible=True)[0]
        kps1, descs1 = features1.keypoints, features1.descriptors

    return kps1, descs1


def process_image(image_path, disk):
    image = cv2.imread(image_path)
    image = white_balance_grayworld(image)
    image = kornia.image_to_tensor(image, keepdim=False)
    image = kornia.color.bgr_to_rgb(image)
    image = K.geometry.resize(image, (640, 480)) / 255
    keypoints, descriptors = extract_keypoints_and_descriptors(image, disk)
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

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()

    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    return good_matches.copy()

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2


def process_match_image(folder_name, image_name, image_data, input_keypoints, input_descriptors, lg_matcher,
                        device='cpu'):
    kps1, descs1 = image_data['keypoints'].to(device), image_data['descriptors'].to(device)

    # Your processing logic here
    # For simplicity, let's just calculate the match count as the number of keypoints

    dists, idxs = lg_matcher(descs1.to(device), input_descriptors.to(device), KF.laf_from_center_scale_ori(kps1[None],
                                                                                     torch.ones(1, len(kps1), 1, 1,
                                                                                                device=device)),
                             KF.laf_from_center_scale_ori(input_keypoints[None],
                                                          torch.ones(1, len(input_keypoints), 1, 1, device=device)))

    mkpts1, mkpts2 = get_matching_keypoints(kps1, input_keypoints, idxs)

    try:
        Fm, inliers = cv2.findFundamentalMat(
            mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(),
        )
        inliers = inliers > 0
    except cv2.error as e:
        print("Error in cv2.findFundamentalMat:", e)
        inliers = dists < 0.8

    match_count = inliers.shape[0]

    return (folder_name, image_name, match_count, 0, inliers, kps1, descs1, idxs)

def process_image_wrapper(args):
    folder_name, image_name, image_data, input_keypoints, input_descriptors, lg_matcher,device = args
    result = process_match_image(folder_name, image_name, image_data, input_keypoints, input_descriptors, lg_matcher,device=device)
    return result


def find_top_matches(input_keypoints, input_descriptors, keypoints_dict, images_folder, lg_matcher, top_x=5,device='cpu'):
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    progress_text = "Finding where you are, hold on tight!"
    current_image_count = 0
    top_matches = []
    progress_text = "Finding your location..."
    progress_bar = st.progress(0, text=progress_text)
    for folder_name, folder_data in keypoints_dict.items():
        for image_name, image_data in folder_data.items():
            result = process_match_image(folder_name, image_name, image_data, input_keypoints, input_descriptors,
                                         lg_matcher,device=device)
            top_matches.append(result)
            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)
            print(f"{progress_text} ({current_image_count}/{total_images})")

    top_matches.sort(key=lambda x: x[2], reverse=True)
    top_matches = top_matches[:top_x] if top_x > 0 else top_matches
    progress_bar.empty()

    return top_matches


def load_image_from_web(uploaded_file):
    input_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    return input_image


def main():
    st.title('The Lost Student ICVL Project')
    with lock:
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    # device = '0'
    device = K.utils.get_cuda_or_mps_device_if_available()
    # device='cpu'
    transform = T.Resize((640, 480))

    print(device)
    with lock:
        disk, lg_matcher = load_model(device=device)
    num_features = 2048
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace("\\", "/")
    print(dir_path)
    images_folder = dir_path + "/The_Lost_Student_App/images"
    images_folder = "D:/University/msc/semester a/intro to computational biology and vision/final_project/The_Lost_Student_App/images"
    images_folder = dir_path + "/images"
    keypoints_file = "keypoints_descriptors.pkl"  # File to save/load keypoints and descriptors

    # Check if keypoints and descriptors file exists, load them if it does
    if os.path.exists(keypoints_file):
        keypoints_dict = load_keypoints_descriptors_from_file(keypoints_file)
    else:
        keypoints_dict = extract_keypoints_descriptors_dict(images_folder)
        # Save keypoints and descriptors to file
        save_keypoints_descriptors_to_file(keypoints_dict, keypoints_file)

    if uploaded_file is not None:
        with lock:
            input_image = load_image_from_web(uploaded_file)
            orig_input_img = input_image.copy()
            hist1 = calculate_color_histogram(orig_input_img)

        st.image(input_image[:, :, ::-1], caption='Uploaded Image', use_column_width=True)
        input_image = white_balance_grayworld(input_image)

        std_size = (640, 480)
        input_image = kornia.image_to_tensor(input_image, keepdim=False)
        input_image = kornia.color.bgr_to_rgb(input_image)
        input_image = transform(input_image) / 255
        with lock:
            input_keypoints, input_descriptors = extract_keypoints_and_descriptors(input_image.to(device), disk=disk)
            top_matches = find_top_matches(input_keypoints, input_descriptors, keypoints_dict, images_folder,
                                           lg_matcher,device=device)
        count_dict = defaultdict(int)
        for idx, (folder_name, image_name, match_count, _, top_match_keypoints, kps1, descs1, idxs) in enumerate(
                top_matches):
            count_dict[folder_name] += 1

            st.write(f"Match {idx + 1}: {folder_name}/{image_name}")
            st.write(f"Number of matches: {match_count}")

            image_path = os.path.join(images_folder, folder_name, image_name)
            match_image = None
            with lock:
                match_image = cv2.imread(image_path)
                orig_match_img = match_image.copy()
                hist2 = calculate_color_histogram(orig_match_img)
            
            match_image = kornia.image_to_tensor(match_image, keepdim=False).to(device=device)
            match_image = kornia.color.bgr_to_rgb(match_image)
            match_image = transform(match_image) / 255

            match_keypoints = keypoints_dict[folder_name][image_name]['keypoints']
            print(top_match_keypoints.shape)
            print(idxs.shape)
            print(KF.laf_from_center_scale_ori(input_keypoints[None].cpu()).shape)
            print(KF.laf_from_center_scale_ori(match_keypoints[None].cpu()).shape)
            with lock:
                draw_LAF_matches(
                    KF.laf_from_center_scale_ori(match_keypoints[None].cpu()),
                    KF.laf_from_center_scale_ori(input_keypoints[None].cpu()),
                    idxs.cpu(),
                    K.tensor_to_image(match_image.cpu()),
                    K.tensor_to_image(input_image.cpu()),
                    top_match_keypoints,
                    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": (1, 1, 0.2, 0.3),
                               "feature_color": None,
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
                st.image(opencv_img[:, :, ::-1], caption=f'Match {idx + 1}: {folder_name}/{image_name}',
                         use_column_width=True)
        most_common_folder = max(count_dict, key=count_dict.get)
        print("THE FOLDER WINNER IS ", most_common_folder)
        st.write("THE FOLDER WINNER IS ", most_common_folder)


if __name__ == "__main__":
    main()
