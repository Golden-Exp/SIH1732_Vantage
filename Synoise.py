import cv2
import numpy as np
import os
import random

#adding dark noise and readout noise to the images
def add_dark_noise(image, mean=0, min_std=5, max_std=50):
    std = random.uniform(min_std, max_std)
    gauss_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_readout_noise(image, mean=0, min_std=5, max_std=50):
    std = random.uniform(min_std, max_std)
    gauss_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_with_dark_noise = add_dark_noise(image)
                image_with_both_noises = add_readout_noise(image_with_dark_noise)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image_with_both_noises)
                print(f"Processed and saved: {output_path}")

input_folder = 'Official Dataset/Synthetic Data/101-200' 
output_folder = 'Official Dataset/Synthetic Data/Noisy 101-200'  

#output
process_images_in_folder(input_folder, output_folder)
