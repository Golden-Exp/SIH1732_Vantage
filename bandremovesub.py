import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('Code/VerticalBanding/chandraaan2-OHRC-BOX-noisy.png', cv2.IMREAD_GRAYSCALE)

column_mean = np.mean(image, axis=0)

banding_pattern = np.tile(column_mean, (image.shape[0], 1))

# Subtract banding pattern
image_subtracted = cv2.subtract(image, banding_pattern.astype(np.uint8))
image_subtracted = cv2.normalize(image_subtracted, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Filtered image after subtraction
plt.subplot(122)
plt.imshow(image_subtracted, cmap='gray')
plt.title('Image After Subtraction')
plt.axis('off')

# Plotting both images side by side
plt.show()

cv2.imwrite('bandremove.jpg', image_subtracted)
