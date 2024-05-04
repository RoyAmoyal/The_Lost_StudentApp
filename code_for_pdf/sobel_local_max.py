import cv2
import numpy as np



def sobel_operator(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator in x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize to 0-255
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return sobelx, sobely, mag


def find_local_maxima(magnitude,suspect_threshold=65, threshold_neighborhood=0.99, neighborhood_size=3):
    local_maxima = []
    # Iterate over each pixel in the magnitude image
    for y in range(magnitude.shape[0]):
        for x in range(magnitude.shape[1]):
            # Consider only white pixels (assuming white corresponds to higher magnitude values)
            if magnitude[y, x] > suspect_threshold:
                # Define neighborhood bounds
                y_min = max(0, y - neighborhood_size // 2)
                y_max = min(magnitude.shape[0], y + neighborhood_size // 2 + 1)
                x_min = max(0, x - neighborhood_size // 2)
                x_max = min(magnitude.shape[1], x + neighborhood_size // 2 + 1)

                # Check if the current pixel is strictly greater than at least one of its neighbors
                if magnitude[y, x] > threshold_neighborhood * np.max(magnitude[y_min:y_max, x_min:x_max]):
                    local_maxima.append((x, y))

    return local_maxima

# Load an image
image = cv2.imread('../images/25/20240416_104918.jpg')
image = cv2.imread('../images/lib_22/20240416_104937.jpg')

# Resize the original image to 1280x720
resized_image = cv2.resize(image, (1280, 720))

# Apply Sobel operator
sobel_x, sobel_y, magnitude = sobel_operator(image)


# Find local maximum points
threshold = 0.99  # Adjust this threshold to control the number of local maximum points
local_maxima = find_local_maxima(magnitude, threshold_neighborhood=threshold)

# Convert the magnitude image to grayscale
magnitude_gray = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

# Mark local maximum points with a red crosshair
for point in local_maxima:
    cv2.drawMarker(magnitude_gray, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=2, thickness=1)

# Resize Sobel X, Sobel Y, and Magnitude images to match the size of the original image
sobel_x_resized = cv2.resize(sobel_x, (1280, 720))
sobel_y_resized = cv2.resize(sobel_y, (1280, 720))
magnitude_resized = cv2.resize(magnitude, (1280, 720))
maximum_resized = cv2.resize(magnitude_gray, (1280, 720))
cv2.imshow("local maxima",maximum_resized)

resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
# Concatenate the images horizontally
concatenated_image = np.concatenate((resized_image, sobel_x_resized, sobel_y_resized, magnitude_resized), axis=1)
concatenated_image = cv2.resize(concatenated_image, (1900, 720))




# Display the concatenated image
cv2.imshow('Original | Sobel X | Sobel Y | Magnitude | local maxima', concatenated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
