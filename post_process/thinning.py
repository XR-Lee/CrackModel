import cv2
import numpy as np

def zhang_suen_thinning(image):
    def thinning_iteration(im, iter):
        marker = np.zeros_like(im)
        for i in range(1, im.shape[0]-1):
            for j in range(1, im.shape[1]-1):
                p2 = im[i-1, j]
                p3 = im[i-1, j+1]
                p4 = im[i, j+1]
                p5 = im[i+1, j+1]
                p6 = im[i+1, j]
                p7 = im[i+1, j-1]
                p8 = im[i, j-1]
                p9 = im[i-1, j-1]
                A  = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                     (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                     (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                     (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
                B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                m1 = (p2 * p4 * p6) if iter == 0 else (p2 * p4 * p8)
                m2 = (p4 * p6 * p8) if iter == 0 else (p2 * p6 * p8)
                if A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0:
                    marker[i, j] = 1
        return im & ~marker

    image = image // 255
    prev = np.zeros_like(image)
    diff = None

    while True:
        image = thinning_iteration(image, 0)
        image = thinning_iteration(image, 1)
        diff = np.sum(np.abs(image - prev))
        prev = image.copy()
        if diff == 0:
            break

    return image * 255

# Read image
input_image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)
if input_image is None:
    raise ValueError("Image not found or path is incorrect")

# Convert to binary
_, binary_image = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)

# Apply thinning
thinned_image = zhang_suen_thinning(binary_image)

# Save or display the result
cv2.imwrite('thinned_image.png', thinned_image)
