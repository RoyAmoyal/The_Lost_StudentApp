import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# def compute_gradient_magnitude_and_orientation(image):
#     # Compute gradient in x and y directions
#     grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#
#     # Compute magnitude and orientation
#     magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
#     orientation = np.arctan2(grad_y, grad_x)
#
#     return magnitude, orientation
#
# def calculate_histogram_of_gradient(orientation, magnitude, bins=9):
#     # Convert orientation to degrees
#     orientation_deg = np.degrees(orientation) % 180
#
#     # Compute histogram of gradients
#     hist, _ = np.histogram(orientation_deg, bins=bins, range=(0, 180), weights=magnitude)
#
#     return hist
#
# def plot_histogram_of_gradient(hist):
#     angles = np.arange(0, 180, 180 // len(hist))
#     plt.figure()
#     plt.bar(angles, hist, width=20)
#     plt.xlabel('Angle')
#     plt.ylabel('Magnitude')
#     plt.title('Histogram of Gradients')
#     plt.show()


def sobel_operator(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator in x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # orientation = np.arctan2(sobely, sobelx)
    orientation = (np.arctan2(sobely, sobelx) * 180 / np.pi) % 360  # Map angles to the range of 0 to 360 degrees

    # Normalize to 0-255
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return sobelx, sobely, mag,orientation


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

def compute_gradient_magnitude_and_orientation(image):
    # Compute gradient in x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = np.arctan2(grad_y, grad_x)

    return magnitude, orientation

def calculate_histogram_of_gradient(orientation, magnitude, bins=9):
    # Convert orientation to degrees
    orientation_deg = (np.degrees(orientation) + 360) % 360
    orientation_deg = (orientation_deg + 6 * (360 / bins)) % 360

    # Compute histogram of gradients
    hist, _ = np.histogram(orientation_deg, bins=bins, range=(0, 360))

    return hist

def plot_histogram_of_gradient(hist, bins=9):
    angles = np.arange(0, 360, 360 / bins)  # Adjust the range to 0-360 degrees
    plt.figure()
    plt.bar(angles, hist, width=360 / bins)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Magnitude')
    plt.title('Histogram of Gradients')
    plt.show()
import matplotlib.patches as patches

def plot_histogram_of_gradient_arrows(hist, bins=9):
    import numpy as np
    import matplotlib.pyplot as plt

    angles = np.arange(0, 360, 360 / bins)  # Adjust the range to 0-360 degrees
    plt.figure()
    bars = plt.bar(angles, hist, width=360 / bins)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Magnitude')
    plt.title('Histogram of Gradients')
    max_hist = max(hist)

    # Add arrows next to the numbers on the x-axis with orientation matching the angle values
    # Customize x-axis ticks to display arrows next to the numbers
    for angle, value in zip(angles, hist):
        dx = np.cos(np.radians(angle)) * 0.5  # Scale the arrow length
        dy = np.sin(np.radians(angle)) * 0.5
        space = max_hist * 0.025
        plt.text(angle, value + space, '\u2192', fontsize=12, ha='center', va='center',
                 rotation=angle)  # Adjust y-coordinate

    plt.xticks((np.arange(0, 360, 360 / bins)), [f'{int(angle)}°' for angle in angles])

    plt.show()


image = cv2.imread('../images/lib_22/20240416_104937.jpg')
image = cv2.imread('../../20240430_093648.jpg')

sobel_x, sobel_y, magnitude,orientation = sobel_operator(image)
local_maxima = find_local_maxima(magnitude)

# Compute gradient magnitude and orientation
# magnitude, orientation = compute_gradient_magnitude_and_orientation(image)

# Randomly select a local maximum point
np.random.seed(42)
local_maxima_indices = np.arange(len(local_maxima))

# Define the neighborhood size
neighborhood_size = 16

# Number of local maxima to select
num_local_maxima = 150
bins = 12
for i in range(num_local_maxima):
    random_local_maxima_index = np.random.choice(local_maxima_indices)

    # Extract the coordinates of the randomly selected local maximum point
    x, y = local_maxima[random_local_maxima_index]
    x ,y= 1550,1300
    print("current x,y ",x," , ",y)
    # Extract the neighborhood around the local maximum point
    neighborhood = magnitude[y - neighborhood_size // 2:y + neighborhood_size // 2,
                              x - neighborhood_size // 2:x + neighborhood_size // 2]

    # Calculate histogram of gradient for the neighborhood around the local maximum point
    hist = calculate_histogram_of_gradient(orientation[y - neighborhood_size // 2:y + neighborhood_size // 2,
                                                       x - neighborhood_size // 2:x + neighborhood_size // 2],
                                            magnitude[y - neighborhood_size // 2:y + neighborhood_size // 2,
                                                       x - neighborhood_size // 2:x + neighborhood_size // 2],bins)

    # Display the location of the selected local maximum point
    plt.figure()
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cmap='gray')
    plt.plot(x, y, 'o', markerfacecolor='red', markeredgecolor='red', markersize=3)
    plt.title('Selected Local Maximum Point')
    plt.show()

    # Plot the histogram of gradient
    # plot_histogram_of_gradient(hist)
    plot_histogram_of_gradient_arrows(hist,bins)



