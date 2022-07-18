import torch
import numpy as np
import os
import config
from models import densenet121, comp_resnet50, comp_dense, densenet_2stage, densenet_3stage, adaptive_densenet
import torch.nn as nn

def load_densenet_model(model, oristate_dict):
    cfg = {'dense121': [6, 12, 24, 16]}

    state_dict = model.state_dict()

    current_cfg = cfg['dense121']
    last_select_index = None
    last_concat_index = None

    all_honey_conv_weight = []

    bn_part_name=['.weight','.bias','.running_mean','.running_var']#,'.num_batches_tracked']
    prefix = 'rank_adaptive_local3/densenet121_limit3/rank_conv'
    subfix = '.npy'
    cnt=1

    conv_weight_name = 'features.conv0.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[conv_weight_name]
    orifilter_num = oriweight.size(0)
    start_filternum = orifilter_num
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        print('loading rank from: ' + prefix + str(cnt) + subfix)
        rank = np.load(prefix + str(cnt) + subfix)
        select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()
        for index_i, i in enumerate(select_index):
            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict['features.norm0' + bn_part][index_i] = oristate_dict['features.norm0' + bn_part][i]
        last_select_index = select_index
    else:
        state_dict[conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict['features.norm0' + bn_part] = oristate_dict['features.norm0'+bn_part]
        # last_select_index = np.array(range(int(64*0.65)))
        # last_select_index = np.array(range(64))
        last_select_index = np.array(range(orifilter_num))

    state_dict['features.norm0' + '.num_batches_tracked'] = oristate_dict['features.norm0' + '.num_batches_tracked']

    model_block = [model.features.denseblock1, model.features.denseblock2, model.features.denseblock3, model.features.denseblock4]
    trainsition_block = [model.features.transition1, model.features.transition2, model.features.transition3]
    cnt+=1
    print(len(last_select_index))
    for layer, num in enumerate(current_cfg):
        if last_select_index is not None:
            last_concat_index = last_select_index
        else : 
            last_concat_index = np.array(range(start_filternum))
            # last_concat_index = np.array(range(int(2**(layer+6))))
            
            # if layer == 0:
            #     last_concat_index = np.array(range(int(64*0.65)))
            # elif layer == 1:
            #     last_concat_index = np.array(range(80))
            # elif layer == 2:
            #     last_concat_index = np.array(range(160))
            # elif layer == 3:
            #     last_concat_index = np.array(range(320))
            
        
        block_name = 'features.' + 'denseblock' + str(layer + 1) + '.'

        ori_out_num = start_filternum

        for k in range(num):
            layer_name = block_name + 'denselayer' + str(k+1) + '.'

            for num_conv in range(1,3):
                if num_conv == 1:
                    last_select_index = last_concat_index
                bn_name = layer_name + 'norm' + str(num_conv)
                conv_name = layer_name + 'conv' + str(num_conv)
                
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                print(cnt, orifilter_num, currentfilter_num)
                # print(conv_weight_name)
                # print(orifilter_num)
                # print(currentfilter_num)
                # print(len(last_select_index))

                if orifilter_num != currentfilter_num:
                    
                    rank = np.load(prefix + str(cnt) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()
                    
                    if last_select_index is not None:
                        
                        print('(input&output)loading rank from: ' + prefix + str(cnt) + subfix)
                        print('input : ', len(last_select_index))
                        print('output : ', len(select_index))
                        # print('select_index : ', select_index)
                        # print('Last_select_index : ', last_select_index)
                        for index_j, j in enumerate(last_select_index):
                            for index_i, i in enumerate(select_index):
                                state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                    else:
                        print('(output)loading rank from: ' + prefix + str(cnt) + subfix)
                        print('input : ', len(oristate_dict[conv_weight_name][0]))
                        print('output : ', len(select_index))
                        # print('select_index : ', select_index)
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

             
                    last_select_index = select_index


                elif last_select_index is not None:
                    print('(input)loading rank from: ' + prefix + str(cnt) + subfix)
                    print('input : ', len(last_select_index))
                    print('output : ', orifilter_num)
                    # print('Last_select_index : ', last_select_index)
                    for index_j, j in enumerate(last_select_index):
                        for index_i in range(orifilter_num):
                            state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][index_i][j]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                  
                    last_select_index = None
            

                else:
                    print('(no_change)loading rank from: ' + prefix + str(cnt) + subfix)
                    print('input : ', len(oristate_dict[conv_weight_name][0]))
                    print('output : ', orifilter_num)
                    state_dict[conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]
                 
                    last_select_index = None


                state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
             
                cnt+=1
                if num_conv ==2 :
                    # if layer == 0:
                    #     ori_out_num = int(64*0.65) + int(32*0.65) *k
                    # elif layer == 1:
                    #     ori_out_num = 80 + int(32*0.65) *k
                    # elif layer == 2:
                    #     ori_out_num = 160 + int(32*0.65) *k
                    # elif layer == 3:
                    #     ori_out_num = 320 + int(32*0.65) *k

                    
                    
                    # ori_out_num = 2**(layer+6)  + 32 * k
                    # print(ori_out_num)
                    
                    if last_select_index is not None:
                        change = last_select_index + ori_out_num
                        last_concat_index = np.concatenate((last_concat_index, change), axis=0 )
                    else :
                        no_change = np.array(range(orifilter_num)) 
                        no_change = no_change + ori_out_num
                        last_concat_index = np.concatenate((last_concat_index, no_change), axis=0 )

                    ori_out_num += orifilter_num 
                



        if layer != 3:
            block_name = 'features.' + 'transition' + str(layer + 1) + '.'
            bn_name = block_name + 'norm' 
            conv_name = block_name + 'conv' 
            conv_weight_name = conv_name + '.weight'
            all_honey_conv_weight.append(conv_weight_name)
            oriweight = oristate_dict[conv_weight_name]
            curweight = state_dict[conv_weight_name]
            orifilter_num = oriweight.size(0)
            start_filternum = orifilter_num
            currentfilter_num = curweight.size(0)
            last_select_index = last_concat_index

            print(cnt, orifilter_num, currentfilter_num)
        

            if orifilter_num != currentfilter_num:
                
                rank = np.load(prefix + str(cnt) + subfix)
                select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    print('(input&output)loading rank from: ' + prefix + str(cnt) + subfix)
                    print('input : ', len(last_select_index))
                    print('output : ', len(select_index))
                    # print('select_index : ', select_index)
                    # print('Last_select_index : ', last_select_index)
                    for index_j, j in enumerate(last_select_index):
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]

                        for bn_part in bn_part_name:
                            state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

                else:
                    print('(output)loading rank from: ' + prefix + str(cnt) + subfix)
                    print('input : ', len(oristate_dict[conv_weight_name][0]))
                    print('output : ', len(select_index))
                    # print('select_index : ', select_index)
                    for index_i, i in enumerate(select_index):
                        state_dict[conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

                last_select_index = select_index


            elif last_select_index is not None:
                print('(input)loading rank from: ' + prefix + str(cnt) + subfix)
                print('input : ', len(last_select_index))
                print('output : ', orifilter_num)
                # print('Last_select_index : ', last_select_index)
                for index_j, j in enumerate(last_select_index):
                    for index_i in range(orifilter_num):
                        state_dict[conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

      
                last_select_index = None
        

            else:
                print('(no_change)loading rank from: ' + prefix + str(cnt) + subfix)
                print('input : ', len(oristate_dict[conv_weight_name][0]))
                print('output : ', orifilter_num)
                state_dict[conv_weight_name] = oriweight
                for bn_part in bn_part_name:
                    state_dict[bn_name + bn_part] = \
                        oristate_dict[bn_name + bn_part]
         
                last_select_index = None


            state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
     
            cnt+=1
            print(len(last_concat_index))

    bn_name = 'features.norm5'  
    last_select_index = last_concat_index
    if last_select_index is not None:
        print('(input)norm5')
        print('norm : ', len(last_select_index))
        for index_j, j in enumerate(last_select_index):
    
            for bn_part in bn_part_name:
                state_dict[bn_name + bn_part][index_j] = oristate_dict[bn_name + bn_part][j]

    else: 
        print('(no)norm5')
        for bn_part in bn_part_name:
            state_dict[bn_name + bn_part] = oristate_dict[bn_name + bn_part]

    state_dict[ bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                print('*******************error********************')
                state_dict[conv_name] = oristate_dict[conv_name]

        if isinstance(module, nn.Linear):
            print('fill fc layer ' + str(name))
            if name == 'fc1':
                
                if last_select_index is not None :
                    
                    new_output = state_dict[name + '.weight'].size(0)
                    orifilter_input = oristate_dict[name + '.weight'][0].size(0)
                    
                    input_num = orifilter_input-157
                    print(input_num)
                    change = np.array(range(157)) + input_num
                    last_concat_index = np.concatenate((last_select_index, change), axis=0 )
                    print('input : ', len(last_concat_index))
                    print('output : ', new_output)
                    for index_j, j in enumerate(last_concat_index):
                        for index_i in range(new_output):
                            state_dict[name + '.weight'][index_i][index_j] = oristate_dict[name + '.weight'][index_i][j]

                    state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                    
                else: 
                    state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                    state_dict[name + '.bias'] = oristate_dict[name + '.bias']
            else:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    torch.save(checkpoint, 'pruned_model/adaptive_local_densenet4' + '.pt' ) 
  
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


compress_rate_2 = [[0.25], [0.375, 0.46875, 0.40625, 0.3125, 0.34375, 0.4375], [0.4375, 0.5, 0.375, 0.4375, 0.34375, 0.4375, 0.4375, 0.4375, 0.5, 0.46875, 0.53125, 0.375], [0.40625, 0.5625, 0.5, 0.46875, 0.40625, 0.375, 0.4375, 0.5625, 0.5, 0.65625, 0.46875, 0.5625, 0.65625, 0.46875, 0.5, 0.5, 0.5625, 0.53125, 0.65625, 0.5, 0.375, 0.5625, 0.5625, 0.53125], [0.53125, 0.5, 0.59375, 0.5, 0.53125, 0.5625, 0.59375, 0.375, 0.5, 0.46875, 0.5, 0.59375, 0.5625, 0.53125, 0.5, 0.53125]]
compress_rate2_2 = [[0], [0.7109375, 0.6171875, 0.5234375, 0.53125, 0.4609375, 0.40625], [0.5234375, 0.4609375, 0.4296875, 0.4765625, 0.3984375, 0.4609375, 0.390625, 0.4609375, 0.46875, 0.4375, 0.3203125, 0.4140625], [0.5078125, 0.4375, 0.4609375, 0.5234375, 0.484375, 0.484375, 0.5546875, 0.4921875, 0.546875, 0.5234375, 0.5, 0.4921875, 0.4765625, 0.53125, 0.515625, 0.4921875, 0.546875, 0.53125, 0.5078125, 0.5078125, 0.5078125, 0.484375, 0.4921875, 0.453125], [0.546875, 0.578125, 0.5546875, 0.53125, 0.5078125, 0.5078125, 0.5234375, 0.484375, 0.515625, 0.5546875, 0.546875, 0.453125, 0.5546875, 0.53125, 0.53125, 0.5546875]]
model = adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate_2, compress_rate2_2).to(device)


compress_rate_2 = [[0.203125], [0.28125, 0.4375, 0.34375, 0.25, 0.25, 0.3125], [0.375, 0.40625, 0.3125, 0.375, 0.28125, 0.375, 0.34375, 0.375, 0.46875, 0.4375, 0.5, 0.3125], [0.3125, 0.5, 0.4375, 0.4375, 0.375, 0.3125, 0.375, 0.5, 0.4375, 0.59375, 0.4375, 0.53125, 0.59375, 0.4375, 0.46875, 0.46875, 0.53125, 0.46875, 0.625, 0.4375, 0.34375, 0.53125, 0.5, 0.5], [0.5, 0.4375, 0.5625, 0.4375, 0.5, 0.5, 0.5, 0.34375, 0.5, 0.4375, 0.5, 0.5625, 0.53125, 0.53125, 0.5, 0.53125]]
compress_rate2_2 = [[0], [0.65625, 0.5859375, 0.4453125, 0.46875, 0.359375, 0.3515625], [0.46875, 0.3984375, 0.359375, 0.4140625, 0.328125, 0.3984375, 0.3203125, 0.3984375, 0.40625, 0.390625, 0.28125, 0.3828125], [0.453125, 0.40625, 0.3984375, 0.4609375, 0.421875, 0.4296875, 0.484375, 0.4296875, 0.4765625, 0.4609375, 0.453125, 0.4296875, 0.421875, 0.484375, 0.453125, 0.4375, 0.484375, 0.4765625, 0.4375, 0.4453125, 0.46875, 0.3984375, 0.421875, 0.3984375], [0.5390625, 0.546875, 0.53125, 0.5, 0.5078125, 0.5078125, 0.515625, 0.484375, 0.515625, 0.5390625, 0.53125, 0.453125, 0.546875, 0.5078125, 0.53125, 0.515625]]
origin_model =  adaptive_densenet(config.SMPL_MEAN_PARAMS, compress_rate_2, compress_rate2_2).to(device)

checkpoint = torch.load('/home/urp10/SPIN/models_trained/adaptive_local_3.pt')

state_dict = checkpoint['model']
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        continue
        # k = 'module.'+k
    else:
        k = k.replace('module.', '')
    new_state_dict[k]=v

origin_model.load_state_dict(new_state_dict)
oristate_dict = origin_model.state_dict()
load_densenet_model(model, oristate_dict)