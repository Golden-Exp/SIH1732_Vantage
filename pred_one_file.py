import os
import yaml
import torch
import numpy as np
import scipy.io as sio
from os.path import join
from model import RDUNet
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from utils import build_ensemble, separate_ensemble, predict_ensemble, mod_pad, mod_crop
from PIL import Image

def predict(model, x, device, padding, n_channels):
    multi_channel = True if n_channels == 3 else False
    x = sio.loadmat(x)['data']
    if padding:
        x, size = mod_pad(x, 8)
    else:
        x = mod_crop(x, 8)
    x = build_ensemble(x, normalize=False)
    print("hi")
    with torch.no_grad():
        y_hat_ens = predict_ensemble(model, x, device)
        y_hat_ens, y_hat = separate_ensemble(y_hat_ens, return_single=True)
    if padding:
        y_hat = y_hat[:size[0], :size[1], ...]
        y_hat_ens = y_hat_ens[:size[0], :size[1], ...]
    #print(peak_signal_noise_ratio(np.array(x), y_hat, data_range=1.))
    y_hat = (255 * y_hat).astype('uint8')
    y_hat_ens = (255 * y_hat_ens).astype('uint8')

    y_hat = np.squeeze(y_hat)
    y_hat_ens = np.squeeze(y_hat_ens)
    
    io.imsave("Output/trial.jpg", y_hat)

    io.imsave("Output/trial2.jpg", y_hat_ens)

with open('config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

model_params = config['model']
test_params = config['test']
n_channels = model_params['channels']

model_path = join(test_params['pretrained models path'], 'model_color.pth' if n_channels == 3 else 'model_gray.pth')

model = RDUNet(**model_params)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("Using device: {}".format(device))

# Convert image to arrays so that you can pass it into the predict
img1 = Image.open('Images/b1.png')
noisy = np.array(img1)
img2 = Image.open('Images/b2.png')
grdtrth = np.array(img2)


state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()
noisy_image_path = "Images/chandraaan2-OHRC-BOX-noisy.mat"
padding = False
predict(model, noisy_image_path, device, padding, n_channels)

