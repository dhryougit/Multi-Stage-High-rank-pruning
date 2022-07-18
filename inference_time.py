import torch
from torch.utils.data import DataLoader
import numpy as np

import os

import config

from models import my_model, densenet121, comp_resnet50, comp_dense, densenet_3stage


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


compress_rate = [0]*21
model1 = comp_resnet50(config.SMPL_MEAN_PARAMS, compress_rate)
model1.eval()
model1.to(device)

compress_rate = [ [0.], [0.]*6, [0.]*12, [0.]*24, [0.]*16 ]
model2 = comp_dense(config.SMPL_MEAN_PARAMS, compress_rate)
model2.eval()
model2.to(device)

compress_rate = [ [0.35], [0.35]*6, [0.35]*12, [0.35]*24, [0.35]*16 ]
model3 = densenet_3stage(config.SMPL_MEAN_PARAMS, compress_rate)
model3.eval()
model3.to(device)



batch_size = 1

dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 500
images = repetitions * batch_size
timings1=np.zeros((repetitions,1))
timings2=np.zeros((repetitions,1))
timings3=np.zeros((repetitions,1))

#GPU-WARM-UP
for _ in range(100):
    _ = model1(dummy_input)
    _ = model2(dummy_input)
    _ = model3(dummy_input)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):

        starter.record()
        _ = model1(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings1[rep] = curr_time


        starter.record()
        _ = model2(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings2[rep] = curr_time


        starter.record()
        _ = model3(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings3[rep] = curr_time



        # print(curr_time)
mean_syn1 = np.sum(timings1) / images
std_syn1 = np.std(timings1)

mean_syn2 = np.sum(timings2) / images
std_syn2 = np.std(timings2)

mean_syn3 = np.sum(timings3) / images
std_syn3 = np.std(timings3)


print('batch size : ', batch_size)
print('model1, mean time per image : ', mean_syn1)
print('model2, mean time per image : ', mean_syn2)
print('model3, mean time per image : ', mean_syn3)