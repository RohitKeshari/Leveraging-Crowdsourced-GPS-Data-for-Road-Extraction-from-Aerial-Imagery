import torch
import os

from framework import Trainer
from utils.datasets import prepare_Beijing_dataset
from networks.unet import Unet, UnetMoreLayers
from networks.dlinknet import DinkNet34, LinkNet34
from networks.resunet import ResUnet, ResUnet1DConv
from networks.deeplabv3plus import DeepLabV3Plus
import cv2    
import numpy as np
#input_channels = "gps_only"
#input_channel_num = 1

#input_channels = "image_only"
input_channel_num = 3

#input_channels = "image_gps"
#input_channel_num = 4

model_PATH='./trained_model/dlink34_image_only.pth' 
device = torch.device('cpu') 
model = DinkNet34(num_channels=input_channel_num) 
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_PATH, map_location=device))   



#Read image

img = cv2.imread('10_40_sat.png')    # numpy.ndarray
im=torch.Tensor(img)

im_v=Variable(im, volatile=False)
pred=model(im_v.view(1,3,1024,1024))
p_im=pred.data
nim=p_im.view(1024,1024).numpy()

norm_image = cv2.normalize(nim, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

norm_image.astype(np.uint8)
cv2.imwrite('10_40_sat_image_only.png',norm_image)
