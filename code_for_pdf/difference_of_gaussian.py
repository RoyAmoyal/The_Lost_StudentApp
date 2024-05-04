import cv2
import numpy as np

def detect_local_maxima_dog(image, num_scales, sigma, threshold):
    # Step 1: Compute Difference of Gaussian images
    dog_images = []
    for i in range(num_scales):
        gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma * (2 ** i))
        gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma * (2 ** (i + 1)))
        dog_images.append(gaussian1 - gaussian2)
        print(dog_images[0])

    # Step 2: Find local maxima in each DoG image
    local_maxima = []
    for dog_image in dog_images:
        local_max = np.zeros_like(dog_image, dtype=bool)
        if len(dog_image.shape) == 2:  # 2D image
            print("what2")
            for y in range(1, dog_image.shape[0] - 1):
                for x in range(1, dog_image.shape[1] - 1):
                    if dog_image[y, x] > threshold and \
                       np.all(dog_image[y, x] > dog_image[y-1:y+2, x-1:x+2]):
                        local_max[y, x] = True
                    elif dog_image[y, x] < -threshold and \
                         np.all(dog_image[y, x] < dog_image[y-1:y+2, x-1:x+2]):
                        local_max[y, x] = True
        elif len(dog_image.shape) == 3:  # 3D image
            print("what3")
            for z in range(1, dog_image.shape[0] - 1):
                for y in range(1, dog_image.shape[1] - 1):
                    for x in range(1, dog_image.shape[2] - 1):
                        if dog_image[z, y, x] > threshold and \
                           np.all(dog_image[z, y, x] > dog_image[z-1:z+2, y-1:y+2, x-1:x+2]):
                            local_max[z, y, x] = True
                        elif dog_image[z, y, x] < -threshold and \
                             np.all(dog_image[z, y, x] < dog_image[z-1:z+2, y-1:y+2, x-1:x+2]):
                            local_max[z, y, x] = True
        local_maxima.append(local_max)

    return local_maxima



# Example usage
if __name__ == "__main__":
    # Load the image
    image = cv2.imread("../images/25/20240416_104918.jpg", cv2.IMREAD_GRAYSCALE)

    # Parameters
    num_scales = 1
    sigma = 1.6
    threshold = 1  # Reduced threshold value

    # Detect local maxima using DoG
    local_maxima = detect_local_maxima_dog(image, num_scales, sigma, threshold)

    # Display or output the detected local maxima
    for i, maxima in enumerate(local_maxima):
        print(maxima)
        print(maxima.dtype)
        maxima = maxima.astype(np.uint8) * 255
        maxima = cv2.resize(maxima,(1280,720))
        # cv2.imshow(f"Local Maxima Scale {i+1}", maxima.astype(np.uint8) * 255)

        cv2.imshow(f"Local Maxima Scale {i + 1}", maxima)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
