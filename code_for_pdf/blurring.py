import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/lib_22/20240416_104937.jpg')

image = cv2.resize(image, (1280, 720))

# Apply average blurring
avg_blur = cv2.blur(image, (9, 9))

# Apply median blurring
median_blur = cv2.medianBlur(image, 9)

# Apply Gaussian blurring
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)

# Concatenate images
concatenated_image = np.concatenate((image, avg_blur, median_blur, gaussian_blur), axis=1)
concatenated_image = cv2.resize(concatenated_image, (1280, 720))

# Display concatenated images
cv2.imshow('Original | Average Blurred | Median Blurred | Gaussian Blurred', concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
