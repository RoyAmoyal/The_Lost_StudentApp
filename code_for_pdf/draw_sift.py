import cv2

# Read the image
image = cv2.imread('../images/35/20240416_104559.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create(contrastThreshold=0.1, edgeThreshold=30)

# Detect keypoints
keypoints = sift.detect(gray, None)

# Draw keypoints on the image
# image_with_keypoints = cv2.drawKeypoints(image,keypoints,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))  # BGR format: red

image_with_keypoints = cv2.resize(image_with_keypoints,(1280,720))
# Show the image with keypoints
cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
