import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table  # EDIT: see deprecation warnings below


number_blocks = [6,12,24,16]
rank_importance = [[], [], [], [], []]
pre = 'rank_densenet121/densenet121_limit3'
total_importance = []
layer4_iomportance = []
compress_num = [[0], [0]*6, [0]*12, [0]*24, [0]*16]
compress_num2 = [[0], [0]*6, [0]*12, [0]*24, [0]*16]
total_data_num = [[0], [0]*6, [0]*12, [0]*24, [0]*16]
total_data_num2 = [[0], [0]*6, [0]*12, [0]*24, [0]*16]
compress_rate = [[0], [0]*6, [0]*12, [0]*24, [0]*16]
compress_rate2 = [[0], [0]*6, [0]*12, [0]*24, [0]*16]

y_max = 56
rank = pre + '/rank_conv%d'%(1) + '.npy'
data = np.load(rank)
mean_rank = data.mean()
var_rank = data.std()
sigma = 0.5
crit = mean_rank - sigma * var_rank
low_num = len(np.where(data <= crit)[0])
compress_num[0][0] = low_num
low_rate = low_num / len(data)
total_data_num[0][0] = len(data)
# importance = mean_rank / y_max
rank_importance[0].append([mean_rank, var_rank, low_num, low_rate])
# compress_rate[0].append(low_rate)
data_lenght = len(data)

total_importance

cnt=1


for i in range(4):
    for j in range(number_blocks[i]):
        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        var_rank = data.std()
        crit = mean_rank - sigma * var_rank
        low_num = len(np.where(data <= crit)[0])
        total_data_num2[i+1][j] = len(data)
        compress_num2[i+1][j] = low_num
        low_rate = low_num / len(data)
        # importance = mean_rank / y_max
        rank_importance[i+1].append([mean_rank, var_rank, low_num, low_rate])
        # compress_rate2[i+1].append(low_rate)

        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        var_rank = data.std()
        crit = mean_rank - sigma * var_rank
        low_num = len(np.where(data <= crit)[0])
        total_data_num[i+1][j] = len(data)
        compress_num[i+1][j] = low_num
        low_rate = low_num / len(data)
        # importance = mean_rank / y_max
        rank_importance[i+1].append([mean_rank, var_rank, low_num, low_rate])
        # compress_rate[i+1].append(low_rate)

      

    y_max = y_max // 2
    if i != 3:
        cnt += 1
        rank = pre + '/rank_conv%d'%(cnt) + '.npy'
        data = np.load(rank)
        mean_rank = data.mean()
        var_rank = data.std()
        crit = mean_rank - sigma * var_rank
        low_num = len(np.where(data <= crit)[0])
        low_rate = low_num / len(data)
        # importance = mean_rank / y_max
        rank_importance[i+1].append([mean_rank, var_rank, low_num, low_rate])

# 2stage or 3stage or 4stage
comp_num = [[6], [3, 6, 2, 5, 6, 5], [3, 4, 3, 4, 5, 5, 4, 3, 7, 7, 6, 5], [5, 4, 5, 6, 5, 3, 4, 6, 5, 5, 5, 4, 6, 5, 5, 6, 4, 3, 5, 5, 4, 7, 6, 4], [5, 4, 6, 4, 5, 6, 6, 5, 4, 6, 4, 6, 6, 5, 7, 4]]
comp_num2 = [[0], [16, 26, 17, 23, 15, 19], [21, 21, 18, 19, 14, 16, 18, 20, 20, 16, 11, 19], [16, 18, 20, 17, 19, 23, 21, 17, 20, 19, 22, 19, 22, 21, 22, 20, 20, 19, 19, 18, 22, 20, 20, 18], [21, 20, 24, 18, 22, 20, 22, 27, 29, 27, 21, 21, 22, 23, 25, 21]]
comp_num3 = [[4], [2, 1, 1, 1, 1, 1], [2, 2, 2, 1, 2, 3, 2, 1, 2, 1, 2, 2], [1, 2, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 0, 0]]
comp_num4 = [[0], [3, 3, 5, 7, 5, 7], [7, 4, 5, 7, 9, 6, 7, 6, 4, 6, 5, 9], [7, 4, 4, 8, 9, 4, 6, 5, 6, 5, 5, 6, 4, 7, 6, 5, 8, 8, 9, 7, 6, 6, 6, 3], [1, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]]
comp_num5 = [[3], [3, 1, 2, 2, 3, 4], [2, 3, 2, 2, 2, 2, 3, 2, 1, 1, 1, 2], [3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1], [1, 2, 1, 2, 1, 2, 3, 1, 0, 1, 0, 1, 1, 0, 0, 0]]
comp_num6 = [[0], [7, 4, 10, 8, 13, 7], [7, 8, 9, 8, 9, 8, 9, 8, 8, 6, 5, 4], [7, 4, 8, 8, 8, 7, 9, 8, 9, 8, 6, 8, 7, 6, 8, 7, 8, 7, 9, 8, 5, 11, 9, 7], [1, 4, 3, 4, 0, 0, 1, 0, 0, 2, 2, 0, 1, 3, 0, 5]]


compress_num[0][0] += comp_num[0][0]
compress_num2[0][0] += comp_num2[0][0]
compress_num[0][0] += comp_num3[0][0]
compress_num2[0][0] += comp_num4[0][0]
compress_num[0][0] += comp_num5[0][0]
compress_num2[0][0] += comp_num6[0][0]

for i in range(4):
    for j in range(number_blocks[i]):
        compress_num[i+1][j] += comp_num[i+1][j]
        compress_num2[i+1][j] += comp_num2[i+1][j]

for i in range(4):
    for j in range(number_blocks[i]):
        compress_num[i+1][j] += comp_num3[i+1][j]
        compress_num2[i+1][j] += comp_num4[i+1][j]

for i in range(4):
    for j in range(number_blocks[i]):
        compress_num[i+1][j] += comp_num5[i+1][j]
        compress_num2[i+1][j] += comp_num6[i+1][j]



compress_rate[0][0] = compress_num[0][0] / total_data_num[0][0]
for i in range(4):
    for j in range(number_blocks[i]):
        compress_rate[i+1][j] = compress_num[i+1][j] / total_data_num[i+1][j]
        compress_rate2[i+1][j] = compress_num2[i+1][j] / total_data_num2[i+1][j]

print(compress_num)
print(compress_num2)
print(compress_rate)
print(compress_rate2)
# print(result)
# print(result2)
