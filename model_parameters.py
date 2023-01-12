import torch
from torch.utils.data import DataLoader
import numpy as np

import os

import config

from models import my_model, densenet121, comp_resnet50, comp_dense, densenet_2stage, densenet_3stage, adaptive_densenet 
# from results import my_model
from thop import profile

def count_parameters_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# compress_rate = [ [0], [0]*6, [0]*12, [0]*24, [0]*16 ]
# compress_rate2 =  [ [0], [0]*6, [0]*12, [0]*24, [0]*16 ]
# model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate, compress_rate2)

compress_rate = [[0.046875], [0.15625, 0.25, 0.15625, 0.0625, 0.0625, 0.125], [0.25, 0.21875, 0.09375, 0.21875, 0.09375, 0.15625, 0.1875, 0.21875, 0.21875, 0.1875, 0.25, 0.28125], [0.25, 0.25, 0.25, 0.21875, 0.25, 0.1875, 0.21875, 0.28125, 0.25, 0.3125, 0.21875, 0.34375, 0.25, 0.21875, 0.34375, 0.28125, 0.28125, 0.375, 0.34375, 0.28125, 0.3125, 0.28125, 0.25, 0.28125], [0.25, 0.21875, 0.3125, 0.375, 0.21875, 0.375, 0.25, 0.3125, 0.28125, 0.3125, 0.375, 0.3125, 0.25, 0.375, 0.25, 0.375]]
compress_rate2 = [[0], [0.4765625, 0.3671875, 0.265625, 0.234375, 0.1796875, 0.15625], [0.2578125, 0.2109375, 0.1875, 0.2265625, 0.140625, 0.25, 0.1796875, 0.171875, 0.2109375, 0.1796875, 0.140625, 0.171875], [0.25, 0.21875, 0.1953125, 0.234375, 0.2265625, 0.21875, 0.265625, 0.28125, 0.2890625, 0.2578125, 0.2578125, 0.2109375, 0.2578125, 0.2265625, 0.265625, 0.2578125, 0.2109375, 0.25, 0.1875, 0.2109375, 0.2421875, 0.21875, 0.3203125, 0.21875], [0.40625, 0.40625, 0.28125, 0.328125, 0.375, 0.328125, 0.328125, 0.3046875, 0.2890625, 0.2890625, 0.28125, 0.3359375, 0.3671875, 0.3203125, 0.3125, 0.34375]]
model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate, compress_rate2).to(device)
# compress_rate = [0] + [0]*4 + [0]*16 
# model = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate).to(device)

# num_trainable_parameters = count_parameters_trainable(model)
# num_parameters = count_parameters(model)

# print('compressed_model : ', num_trainable_parameters)
# print('compressed_model : ', num_parameters)


# model = hmr(config.SMPL_MEAN_PARAMS)
# num_trainable_parameters = count_parameters_trainable(model)
# num_parameters = count_parameters(model)

# print('original_model : ', num_trainable_parameters)
# print('original_model : ', num_parameters)

input_image_size = 224
input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
flops, params = profile(model, inputs=(input_image,))


print('Params: %.2f'%(params))
print('Flops: %.2f'%(flops))
