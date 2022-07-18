import numpy as np
import matplotlib.pyplot as plt



number_blocks = [6,12,24,16]

pre = 'rank_adaptive_local2/densenet121_limit3'
rank_visual = 'rank_visual_adaptive_local2'
rank_Type = 'rank'

if rank_Type == 'rank':
    y_max = 56
    rank = pre + '/rank_conv%d'%(1) + '.npy'
    data = np.load(rank)
    plt.plot(data , 'ro')
    plt.title( 'conv%d'%(0) + '_rank', fontsize=15 ) 
    plt.ylim([-1, y_max+1]) 
    plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
    # plt.xscale("log")
    plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
    plt.savefig(rank_visual + '/conv%d'%(1) + '_rank')
    plt.close()

    cnt=1

    for i in range(4):
        for j in range(number_blocks[i]):
            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'block%d'%(i+1) + '-layer%d'%(j+1) +'_1x1' + '_rank' ,fontsize=15) 
            plt.ylim([-1, y_max+1]) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'block%d'%(i+1) + '-layer%d'%(j+1) +'_3x3' + '_rank' ,fontsize=15 ) 
            plt.ylim([-1, y_max+1]) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

        y_max = y_max // 2
        if i != 3:
            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'trainsition%d'%(i+1) +  '_rank' ,fontsize=15) 
            plt.ylim([-1, y_max+1]) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

else : 
    rank = pre + '/rank_conv%d'%(1) + '.npy'
    data = np.load(rank)
    plt.plot(data , 'ro')
    plt.title( 'conv%d'%(0) + '_rank',fontsize=15 ) 
    plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
    # plt.xscale("log")
    plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
    plt.savefig(rank_visual + '/conv%d'%(1) + '_rank')
    plt.close()

    cnt=1

    for i in range(4):
        for j in range(number_blocks[i]):
            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'block%d'%(i+1) + '-layer%d'%(j+1) +'_1x1' + '_rank',fontsize=15 ) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'block%d'%(i+1) + '-layer%d'%(j+1) +'_3x3' + '_rank' ,fontsize=15 ) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

        if i != 3:
            cnt += 1
            rank = pre + '/rank_conv%d'%(cnt) + '.npy'
            data = np.load(rank)
            plt.plot(data , 'ro')
            plt.title( 'trainsition%d'%(i+1) +  '_rank',fontsize=15 ) 
            plt.xlabel('featrue maps', fontsize=15) # x축 label : 'Students_num'
            # plt.xscale("log")
            plt.ylabel('rank', fontsize=15) # y축 label : 'Score'
            plt.savefig(rank_visual + '/rank_conv%d'%(cnt))
            plt.close()

