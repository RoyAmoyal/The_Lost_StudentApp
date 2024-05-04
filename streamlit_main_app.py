import concurrent
import concurrent.futures
import io

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

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from pathlib import Path
from lightglue.utils import load_image, rbd,read_image,numpy_image_to_torch
from lightglue import viz2d
from skimage import exposure
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Apply histogram equalization to both images

import torch
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



lock = Lock()

from geopy.geocoders import Nominatim
import requests
import folium
import pandas as pd



def get_lat_long_from_address(address):
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode(address)
    return location.latitude, location.longitude

def get_directions_response(lat1, long1, lat2, long2, mode='walk'):
    url = "https://route-and-directions.p.rapidapi.com/v1/routing"
    key = "ee94d00befmsh2664d60a9970a41p1c7064jsn5dcdba27f65f"
    host = "route-and-directions.p.rapidapi.com"
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
    querystring = {"waypoints": f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}", "mode": mode}
    response = requests.request("GET", url, headers=headers, params=querystring)
    return response

def create_map(response):
    # use the response
    mls = response.json()['features'][0]['geometry']['coordinates']
    points = [(i[1], i[0]) for i in mls[0]]
    m = folium.Map()
    kw = {"prefix": "fa", "color": "red", "icon": "arrow-up"}
    angle = 180
    icon = folium.Icon(angle=angle, **kw)
    folium.Marker(points[0], icon=icon, tooltip=str(angle)).add_to(m)
    kw = {"prefix": "fa", "color": "green", "icon": "arrow-up"}
    angle = 180
    icon = folium.Icon(angle=angle, **kw)
    folium.Marker(points[-1], icon=icon, tooltip=str(angle)).add_to(m)
    # add the lines
    folium.PolyLine(points, weight=5, opacity=1).add_to(m)
    # create optimal zoom
    df = pd.DataFrame(mls[0]).rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]
    sw = df[['Lat', 'Lon']].min().values.tolist()
    ne = df[['Lat', 'Lon']].max().values.tolist()
    m.fit_bounds([sw, ne])
    return m

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


# @st.cache
def load_model(device='cpu'):
    extractor = SIFT(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="sift").eval().to(device)
    return extractor, matcher


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
# @st.cache_data
def save_keypoints_descriptors_to_file(keypoints_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(keypoints_dict, f)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# Function to load keypoints and descriptors from file
@st.cache_data(max_entries=1)
def load_keypoints_descriptors_from_file(file_path):
    with open(file_path, 'rb') as f:
        # keypoints_dict = CPU_Unpickler(f).load()
        keypoints_dict = pickle.load(f)
    return keypoints_dict


def extract_keypoints_and_descriptors(img, extractor, num_features=2048):
    with torch.inference_mode():
        features1 = extractor(img, num_features, pad_if_not_divisible=True)[0]
        kps1, descs1 = features1.keypoints, features1.descriptors

    return kps1, descs1


def extract_sift(img, extractor, device):
    feats0 = extractor.extract(img.to(device))
    return feats0


def process_image(image_path, disk):
    image = cv2.imread(image_path)
    image = white_balance_grayworld(image)
    image = kornia.image_to_tensor(image, keepdim=False)
    image = kornia.color.bgr_to_rgb(image)
    image = K.geometry.resize(image, (640, 480)) / 255
    keypoints, descriptors = extract_keypoints_and_descriptors(image, disk)
    return keypoints, descriptors


def extract_keypoints_descriptors_dict(images_folder, extractor, device):
    keypoints_dict = {}
    progress_text = "Doing some heavy lifting know so you won't have to wait later."
    progress_bar = st.progress(0, text=progress_text)
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    current_image_count = 0
    idx = 0
    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        folder_data = {}

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # img = load_image(Path(image_path)).cpu().numpy()
            img = read_image(Path(image_path))

            img = cv2.resize(img,(640,480))
            # img = white_balance_grayworld(img)
            # img = exposure.equalize_hist(img)
            img = numpy_image_to_torch(img).to(device)
            # img = torch.from_numpy(img).to(device)

            feats0 = extract_sift(img, extractor, device)
            folder_data[image_name] = feats0
            idx += 1
            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)

        keypoints_dict[folder_name] = folder_data

    progress_bar.empty()
    return keypoints_dict

def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()

    # bf = cv.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))
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


def process_match_image(folder_name, image_name, image_data, pred, lg_matcher,
                        device='cpu'):
    matches01 = lg_matcher({"image0": image_data, "image1": pred})
    feats0, feats1, matches01 = [
        rbd(x) for x in [image_data, pred, matches01]
    ]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    if m_kpts0 is not None and m_kpts0.detach().cpu().numpy().shape[0] > 7:
        try:
            Fm, inliers = cv2.findFundamentalMat(
                m_kpts0.detach().cpu().numpy(), m_kpts1.detach().cpu().numpy(), cv2.FM_RANSAC,
                ransacReprojThreshold=0.5, confidence=0.99
            )
            matches = inliers > 0

        except cv2.error as e:
            print("Error in cv2.findFundamentalMat:", e)

    match_count = matches.shape[0]

    return (folder_name, image_name, match_count, 0, m_kpts0, m_kpts1, matches)



def process_image_wrapper(args):
    folder_name, image_name, image_data, input_keypoints, input_descriptors, lg_matcher, device = args
    result = process_match_image(folder_name, image_name, image_data, input_keypoints, input_descriptors, lg_matcher,
                                 device=device)
    return result


def find_top_matches(pred, keypoints_dict, images_folder, lg_matcher, top_x=5, device='cpu'):
    total_images = sum(len(files) for _, _, files in os.walk(images_folder))
    progress_text = "Finding where you are, hold on tight!"
    current_image_count = 0
    top_matches = []
    progress_text = "Finding your location..."
    progress_bar = st.progress(0, text=progress_text)
    for folder_name, folder_data in keypoints_dict.items():
        for image_name, image_data in folder_data.items():
            result = process_match_image(folder_name, image_name, image_data, pred,
                                         lg_matcher, device=device)

            top_matches.append(result)
            current_image_count += 1
            progress_bar.progress(current_image_count / total_images, text=progress_text)
            print(f"{progress_text} ({current_image_count}/{total_images})")

    top_matches.sort(key=lambda x: x[2], reverse=True)
    top_matches = top_matches[:top_x] if top_x > 0 else top_matches
    progress_bar.empty()

    return top_matches


# @st.cache_data
def load_image_from_web(uploaded_file):
    input_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    input_image = cv2.resize(input_image,(640,480))
    return input_image


def is_similar(image1, image2):
    if image1 is None or image2 is None:
        return False
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())


def handle_destination_map(src_location):
    from geopy.geocoders import Nominatim
    import requests
    import folium
    import pandas as pd

    locations = {'25': [31.26168698113933, 34.80166743349586],
                 '35': [31.261743260802724, 34.80404991966534],
                 '37': [31.262244016475712, 34.80398969323603],
                 '42': [31.262282558939823, 34.80537674427964],
                 '97': [31.26422049296359, 34.80166218505895],
                 'lib_22': [31.262006833782113, 34.80103086668221],
                 'mexico': [31.26250825970978, 34.80571853121805]
                 }

    def get_lat_long_from_address(address):
        locator = Nominatim(user_agent='myGeocoder')
        location = locator.geocode(address)
        return location.latitude, location.longitude

    def get_directions_response(lat1, long1, lat2, long2, mode='walk'):
        url = "https://route-and-directions.p.rapidapi.com/v1/routing"
        key = "ee94d00befmsh2664d60a9970a41p1c7064jsn5dcdba27f65f"
        host = "route-and-directions.p.rapidapi.com"
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
        querystring = {"waypoints": f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}", "mode": mode}
        response = requests.request("GET", url, headers=headers, params=querystring)
        return response

    def create_map(response):
        # use the response
        mls = response.json()['features'][0]['geometry']['coordinates']
        points = [(i[1], i[0]) for i in mls[0]]
        m = folium.Map()
        kw = {"prefix": "fa", "color": "red", "icon": "arrow-up"}
        angle = 180
        icon = folium.Icon(angle=angle, **kw)
        folium.Marker(points[0], icon=icon, tooltip=str(angle)).add_to(m)
        kw = {"prefix": "fa", "color": "green", "icon": "arrow-up"}
        angle = 180
        icon = folium.Icon(angle=angle, **kw)
        folium.Marker(points[-1], icon=icon, tooltip=str(angle)).add_to(m)
        # add the lines
        folium.PolyLine(points, weight=5, opacity=1).add_to(m)
        # create optimal zoom
        df = pd.DataFrame(mls[0]).rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]
        sw = df[['Lat', 'Lon']].min().values.tolist()
        ne = df[['Lat', 'Lon']].max().values.tolist()
        m.fit_bounds([sw, ne])
        return m

    # Add a text input field for the user to enter the destination number
    # destination = st.text_input("Enter Destination Number")


@st.cache_data
def process_and_match_image(uploaded_file, _extractor, _keypoints_dict, images_folder, _lg_matcher, device):
    with lock:
        input_image = load_image_from_web(uploaded_file)
        input_image_orig = input_image.copy()
        uploaded_file = None

    data_vis_images = []
    if not is_similar(input_image_orig, st.session_state.input_image_old):
        st.session_state.input_image_old = input_image_orig

        st.image(input_image[:, :, ::-1], caption='Uploaded Image', use_column_width=True)
        # input_image = white_balance_grayworld(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        print(input_image.shape)
        input_image = input_image.transpose((2, 0, 1))
        input_image = torch.tensor(input_image / 255.0, dtype=torch.float).cpu().numpy()
        input_image = torch.from_numpy(input_image).to(device)

        print(input_image.shape)

        std_size = (640, 480)

        with lock:
            pred = extract_sift(input_image.to(device), extractor=_extractor, device=device)
            top_matches = find_top_matches(pred, _keypoints_dict, images_folder,
                                           _lg_matcher, device=device)
        count_dict = defaultdict(int)
        number_of_vis = 0
        for idx, (folder_name, image_name, match_count, _, m_kpts0, m_kpts1, matches) in enumerate(
                top_matches):
            count_dict[folder_name] += 1

            image_path = os.path.join(images_folder, folder_name, image_name)
            match_image = None
            with lock:
                match_image = load_image(Path(image_path))


            with lock:
                print(match_image.shape)
                print(input_image.squeeze(0).shape)
                if number_of_vis < 1:

                    st.write(f"Match {idx + 1}: {folder_name}/{image_name}")
                    st.write(f"Number of matches: {match_count}")

                    axes = viz2d.plot_images([match_image, input_image])
                    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    # Convert the buffer to a numpy array
                    buffer_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    plt.close()  # Close the matplotlib plot to free resources

                    # Convert the numpy array to an OpenCV image
                    opencv_img = cv2.imdecode(buffer_img, 1)
                    # img_matches = cv.drawMatches(input_image, input_keypoints, match_image, match_keypoints, top_match_keypoints, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    st.image(opencv_img[:, :, ::-1], caption=f'Match {idx + 1}: {folder_name}/{image_name}',
                             use_column_width=True)
                    data_vis_images.append([idx,folder_name,image_name])
                    number_of_vis += 1
        st.session_state.vis_images = data_vis_images
        st.session_state.src_location = max(count_dict, key=count_dict.get)
        data_vis_images = None
        top_matches = None
        count_dict = None
    else:
        for idx,folder_name,image_name in st.session_state.vis_images:
            image_path = os.path.join(images_folder, folder_name, image_name)
            image= cv2.imread(image_path)
            st.image(image[:, :, ::-1], caption=f'Match {idx + 1}: {folder_name}/{image_name}',
                     use_column_width=True)
        return st.session_state.src_location
    return st.session_state.src_location

src_location = None
def main():
    uploaded_file = None
    st.title('The Lost Student ICVL Project')
    with lock:
        if uploaded_file is None:
            # Delete all the items in Session state
            for key in st.session_state.keys():
                del st.session_state[key]
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if "input_image_old" not in st.session_state:
            st.session_state.input_image_old = None

    destination_input = False
    edit_mode = False

    # device = '0'
    device = K.utils.get_cuda_or_mps_device_if_available()
    # device='cpu'
    transform = T.Resize((640, 480))

    print(device)
    with lock:
        extractor, lg_matcher = load_model(device=device)
    num_features = 2048
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace("\\", "/")
    images_folder = dir_path + "/The_Lost_Student_App/images"
    images_folder = "D:/University/msc/semester a/intro to computational biology and vision/final_project/The_Lost_Student_App/images"
    images_folder = dir_path + "/images"
    keypoints_file = "keypoints_descriptors.pkl"

    if 'keypoints' not in st.session_state:
        if os.path.exists(keypoints_file):
            keypoints_dict = load_keypoints_descriptors_from_file(keypoints_file)
        else:
            keypoints_dict = extract_keypoints_descriptors_dict(images_folder, extractor, device)
            save_keypoints_descriptors_to_file(keypoints_dict, keypoints_file)
        st.session_state.keypoints = keypoints_dict
    keypoints_dict = st.session_state.keypoints

    if 'locations' not in st.session_state:
        # uploaded_file = True
        locations = {'25': [31.26168698113933, 34.80166743349586],
                     '35': [31.261743260802724, 34.80404991966534],
                     '37': [31.262244016475712, 34.80398969323603],
                     '97': [31.26422049296359, 34.80166218505895],
                     'lib_22': [31.262006833782113, 34.80103086668221],
                     }
        st.session_state.locations = locations
    destination = st.selectbox("Enter Destination Number", (st.session_state.locations.keys()))

    if destination and uploaded_file is not None:
        src_location = process_and_match_image(uploaded_file, extractor, keypoints_dict, images_folder, lg_matcher,
                                               device)



        if src_location:
            print("Your Location is: ", src_location)
            st.write("Your Location is: ", src_location)
            response = get_directions_response(*st.session_state.locations[src_location], *st.session_state.locations[destination])
            m = create_map(response)
            st.components.v1.html(m._repr_html_(), width=800, height=500)


if __name__ == "__main__":
    main()
