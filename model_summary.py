import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import os

import config

from models import  densenet121, comp_resnet50, comp_dense, adaptive_densenet

from torchinfo import summary

# compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
# model = comp_dense(config.SMPL_MEAN_PARAMS, compress_rate)

compress_rate_2 = [[0.234375], [0.34375, 0.46875, 0.40625, 0.28125, 0.3125, 0.3125], [0.4375, 0.4375, 0.375, 0.40625, 0.34375, 0.4375, 0.375, 0.40625, 0.5, 0.46875, 0.53125, 0.375], [0.34375, 0.5625, 0.4375, 0.4375, 0.40625, 0.3125, 0.40625, 0.53125, 0.5, 0.625, 0.46875, 0.5625, 0.59375, 0.4375, 0.5, 0.46875, 0.53125, 0.46875, 0.65625, 0.4375, 0.375, 0.5625, 0.53125, 0.53125], [0.53125, 0.4375, 0.5625, 0.4375, 0.5, 0.5, 0.5, 0.375, 0.5, 0.4375, 0.5, 0.59375, 0.5625, 0.53125, 0.5, 0.53125]]
compress_rate2_2 = [[0], [0.65625, 0.59375, 0.484375, 0.5, 0.40625, 0.390625], [0.5078125, 0.421875, 0.390625, 0.4453125, 0.375, 0.4453125, 0.359375, 0.4296875, 0.4375, 0.421875, 0.3125, 0.4140625], [0.4765625, 0.4375, 0.4453125, 0.5, 0.453125, 0.453125, 0.515625, 0.4609375, 0.5, 0.484375, 0.4921875, 0.4765625, 0.453125, 0.5, 0.484375, 0.4765625, 0.515625, 0.5078125, 0.4453125, 0.484375, 0.4921875, 0.4375, 0.453125, 0.421875], [0.5390625, 0.546875, 0.53125, 0.5, 0.5078125, 0.5078125, 0.515625, 0.484375, 0.515625, 0.5390625, 0.53125, 0.453125, 0.546875, 0.5078125, 0.53125, 0.515625]]

model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate_2, compress_rate2_2)
# compress_rate = [ [0], [0]*6, [0]*12, [0]*24, [0]*16 ]
# compress_rate2 =  [ [0], [0]*6, [0]*12, [0]*24, [0]*16 ]
# model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate, compress_rate2)

summary(model, (1, 3, 224, 224))
# state_dict = model.state_dict()
# for key in enumerate(state_dict.keys()):
#     print(key)
# print(state_dict['fc1.bias'].shape)

