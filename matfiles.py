from PIL import Image
import numpy as np
from skimage import io
import scipy.io as sio
img = Image.open("images/moon.png")
img = img.resize((256, 256))
img.save("images/moon_resize.png")

img = io.imread('images/moon_resize.png')
# Image has 4 channels. Convert to 1 channel (greyscale)
img = img[:,:,0]
img = img / 255
sio.savemat('images/moon.mat', {'data': img})


