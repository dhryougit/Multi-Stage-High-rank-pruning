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

compress_rate_2 = [[0.25], [0.375, 0.46875, 0.40625, 0.3125, 0.34375, 0.4375], [0.4375, 0.5, 0.375, 0.4375, 0.34375, 0.4375, 0.4375, 0.4375, 0.5, 0.46875, 0.53125, 0.375], [0.40625, 0.5625, 0.5, 0.46875, 0.40625, 0.375, 0.4375, 0.5625, 0.5, 0.65625, 0.46875, 0.5625, 0.65625, 0.46875, 0.5, 0.5, 0.5625, 0.53125, 0.65625, 0.5, 0.375, 0.5625, 0.5625, 0.53125], [0.53125, 0.5, 0.59375, 0.5, 0.53125, 0.5625, 0.59375, 0.375, 0.5, 0.46875, 0.5, 0.59375, 0.5625, 0.53125, 0.5, 0.53125]]
compress_rate2_2 = [[0], [0.7109375, 0.6171875, 0.5234375, 0.53125, 0.4609375, 0.40625], [0.5234375, 0.4609375, 0.4296875, 0.4765625, 0.3984375, 0.4609375, 0.390625, 0.4609375, 0.46875, 0.4375, 0.3203125, 0.4140625], [0.5078125, 0.4375, 0.4609375, 0.5234375, 0.484375, 0.484375, 0.5546875, 0.4921875, 0.546875, 0.5234375, 0.5, 0.4921875, 0.4765625, 0.53125, 0.515625, 0.4921875, 0.546875, 0.53125, 0.5078125, 0.5078125, 0.5078125, 0.484375, 0.4921875, 0.453125], [0.546875, 0.578125, 0.5546875, 0.53125, 0.5078125, 0.5078125, 0.5234375, 0.484375, 0.515625, 0.5546875, 0.546875, 0.453125, 0.5546875, 0.53125, 0.53125, 0.5546875]]
model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate_2, compress_rate2_2).to(device)
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
