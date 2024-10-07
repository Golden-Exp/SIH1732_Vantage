# Enhancement of Permanently Shadowed Regions in Moon Images using Unet
## Introduction
We have carefully developed and designed a low light enhancement software specifically for generating first of its kind PSR image map of lunar poles captured by OHRC of Chandrayaan-2. This software 
- Has A Custom Trained Residual Dense U-Net Neural Network for Image Denoising
- Enables Users  to vary multiple Image Parameters using Digital Image Processing Techniques for further enhancement
- Masking Tools that helps to make localised changes in the region selected by the user using a brush tool
- Cropping Tools to focus more on the regions of interest
- Provides a flexible and convenient UI to prioritise User’s comfort while using the Software

## Dataset
Readout and Dark Noise applied to artificially rendered moon surface images. We are adding other noises in the next stages and training our model to enhance the permanently shadowed regions of the moon better.  
You can access the link to the synthetic dataset created by adding readout and dark noise of noise levels in the range (5,50) to realistic renders of rocky lunar landscapes[here](https://github.com/issaczerubbabela/SIH1732_Vantage/tree/main/Dataset).  
We are hoping to collect Real dark Frames from PSRs added to well Sun Lit or Transiently Lit Images of the CH2 OHRC followed by the addition of Synthetic Noises similar to the noise profile present in the PSRs.



## Architecture
![[Figs/Picture1.jpg]]

## Ground Truth Image vs. Resultant Image
//Resultant PSNR

## Vertical Band Removal
//before after
