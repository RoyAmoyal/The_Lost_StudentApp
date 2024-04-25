import cv2
import kornia
import kornia as K
import kornia.feature as KF
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *

import cv2


# Assuming kps1, kps2, idxs, img1, img2, and inliers are provided by the caller

matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

device = K.utils.get_cuda_or_mps_device_if_available()

print(device)

# %%capture
fname1 = "what.jpg"
fname2 = "images/lib_22/20240416_104943.jpg"
image = cv2.imread(fname2)
image = kornia.image_to_tensor(image,keepdim=False)
image = kornia.color.bgr_to_rgb(image)
image = K.geometry.resize(image, (640, 480))
lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)


img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=device)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32, device=device)[None, ...]
# img2 = K.geometry.resize(img2,(img1.shape[2],img1.shape[3]))

img1 = K.geometry.resize(img1,(640,480))
img2 = K.geometry.resize(img2,(640,480))
img2=image/255
print(img2.shape)
print(img1.shape)
num_features = 2048
disk = KF.DISK.from_pretrained("depth").to(device)
features1 = disk(img1, num_features, pad_if_not_divisible=True)

hw1 = torch.tensor(img1.shape[2:], device=device)
hw2 = torch.tensor(img2.shape[2:], device=device)
print(img1.shape)

with torch.inference_mode():
    inp = torch.cat([img1, img2], dim=0)
    print(inp.shape)
    features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
    kps1, descs1 = features1.keypoints, features1.descriptors
    kps2, descs2 = features2.keypoints, features2.descriptors
    lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
    lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
    dists, idxs = lg_matcher(descs1, descs2, lafs1, lafs2)


    def get_matching_keypoints(kp1, kp2, idxs):
        mkpts1 = kp1[idxs[:, 0]]
        mkpts2 = kp2[idxs[:, 1]]
        return mkpts1, mkpts2


    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

    Fm, inliers = cv2.findFundamentalMat(
        mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    inliers = inliers > 0
    print(f"{inliers.sum()} inliers with DISK")

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(kps1[None].cpu()),
        KF.laf_from_center_scale_ori(kps2[None].cpu()),
        idxs.cpu(),
        K.tensor_to_image(img1.cpu()),
        K.tensor_to_image(img2.cpu()),
        inliers,
        draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": (1, 1, 0.2, 0.3), "feature_color": None,
                   "vertical": False},ax=ax,return_fig_ax=True
    )
    from io import BytesIO

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
    cv2.imshow("w",opencv_img)
    cv2.waitKey(0)
    # plt.imshow(img)
    plt.show()

    plt.savefig('result.png')

    # img_np = np.array(img)
    # plt.imshow(img_np)
    # plt.axis('off')
    # plt.show()
print(f"{idxs.shape[0]} tentative matches with DISK LightGlue")

