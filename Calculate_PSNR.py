#write a python code to take a ground truth image and an image with noise and calculate the psnr value between the two monochrome images.Calculate other metrics like SSIM, MSE, etc. as well.

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from PIL import Image

def calculate_psnr(noisy_image_path, ground_truth_image_path):
    # Load the noisy image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Load the ground truth image
    ground_truth_image = cv2.imread(ground_truth_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(noisy_image, ground_truth_image, data_range=255)
    
    return psnr

def calculate_ssim(noisy_image_path, ground_truth_image_path):
    # Load the noisy image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Load the ground truth image
    ground_truth_image = cv2.imread(ground_truth_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate SSIM
    ssim = structural_similarity(noisy_image, ground_truth_image, data_range=255)
    
    return ssim

def calculate_mse(noisy_image_path, ground_truth_image_path):
    # Load the noisy image
    noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Load the ground truth image
    ground_truth_image = cv2.imread(ground_truth_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate MSE
    mse = mean_squared_error(noisy_image, ground_truth_image)
    
    return mse

# Example usage
noisy_image_path = "Images/b2.png"
ground_truth_image_path = "Images/b1.png"

noisy_image = Image.open(noisy_image_path)

# code to make the ground truth image have the same resolution as the noisy image
ground_truth_image = Image.open(ground_truth_image_path)
ground_truth_image = ground_truth_image.resize((noisy_image.size[0], noisy_image.size[1]))
ground_truth_image.save("Images/b1.png")

psnr = calculate_psnr(noisy_image_path, ground_truth_image_path)
print("PSNR:", psnr)

ssim = calculate_ssim(noisy_image_path, ground_truth_image_path)
print("SSIM:", ssim)

mse = calculate_mse(noisy_image_path, ground_truth_image_path)
print("MSE:", mse)

# Load the noisy image
img1 = Image.open(noisy_image_path)
noisy = np.array(img1)
# Load the ground truth image
img2 = Image.open(ground_truth_image_path)
grdtrth = np.array(img2)

# Calculate PSNR
psnr = peak_signal_noise_ratio(noisy, grdtrth, data_range=255)
print("PSNR:", psnr)

# Calculate SSIM
ssim = structural_similarity(noisy, grdtrth, data_range=255)
print("SSIM:", ssim)

# Calculate MSE
mse = mean_squared_error(noisy, grdtrth)
print("MSE:", mse)

# The PSNR, SSIM, and MSE values can be used to evaluate the quality of the denoised image compared to the ground truth image. Lower MSE values indicate better image quality, while higher PSNR and SSIM values indicate better image quality.
# The PSNR value is a measure of the peak signal-to-noise ratio, which quantifies the quality of the denoised image compared to the ground truth image. Higher PSNR values indicate better image quality.
# The SSIM value is a measure of the structural similarity between the denoised image and the ground truth image. Higher SSIM values indicate better image quality.
# The MSE value is a measure of the mean squared error between the denoised image and the ground truth image. Lower MSE values indicate better image quality.