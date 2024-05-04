import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Define global variables to store selected point and histogram
selected_point = None
histogram_bins = 12

def on_image_click(event):
    global selected_point
    if event.button == 1:
        selected_point = (int(event.xdata), int(event.ydata))
        selected_point = (1530  ,  1320)
def calculate_histogram_and_show(image, point):
    x, y = point
    neighborhood_size = 16

    # Extract the neighborhood around the selected point
    neighborhood = image[y - neighborhood_size // 2:y + neighborhood_size // 2,
                          x - neighborhood_size // 2:x + neighborhood_size // 2]

    # Convert the neighborhood to grayscale
    neighborhood_gray = cv2.cvtColor(neighborhood, cv2.COLOR_BGR2GRAY)

    # Compute gradient magnitude and orientation
    sobelx = cv2.Sobel(neighborhood_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(neighborhood_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    orientation = (np.arctan2(sobely, sobelx) * 180 / np.pi) % 360

    # Calculate histogram of gradients
    hist, _ = np.histogram(orientation, bins=histogram_bins, range=(0, 360))

    # Plot the neighborhood and the histogram
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(neighborhood, cv2.COLOR_BGR2RGB))
    ax1.set_title('Selected Neighborhood')
    ax1.axis('off')

    angles = np.arange(0, 360, 360 / histogram_bins)
    ax2.bar(angles, hist, width=360 / histogram_bins)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Histogram of Gradients')
    plt.show()

# Load the image
image = cv2.imread('../../20240430_093648.jpg')

# Display the image and set up the click event handler
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Click on the image to select a point')
plt.axis('off')
plt.connect('button_press_event', on_image_click)
plt.show()

# Wait for a point to be selected
while selected_point is None:
    plt.pause(0.1)

# Calculate histogram and display
calculate_histogram_and_show(image, selected_point)
