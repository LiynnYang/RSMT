# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/11/23 19:52
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : run.py.py
# @env         : PyCharm
# @Description :
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.gnn_merge import Actor
from main_flow import main_flow
import argparse

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--degree', type=int, default=4000, help='maximum degree of nets')
parser.add_argument('--direction', type=int, default=0, help='the start point of generate blocks')

args = parser.parse_args()

device = torch.device("cuda:0")


if __name__ == '__main__':
    print('loading the model ....')
    model = Actor().to(device)
    best_ckp_dir = 'save/degree40_GCN_DDP_relu_glimpse/rsmt40b.pt'
    checkpoint = torch.load(best_ckp_dir)
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    print('load the model parameter success!, Start to get local optimial graph.')
    # 1.读取引脚数据，生成局部最优RSMT块，并获取block邻接表

    fig = plt.figure(1, (15, 15))

    sum_length0 = []
    all_time = []
    error = []
    degree = args.degree
    direction = args.direction
    arr = np.random.rand(degree, 2)
    # np.save('test.npy', arr)
    # arr = np.load('test.npy')
    # arr = np.loadtxt('input500.txt')

    # coo_file = '/home/hrm/rsmt/benchmark/n2676.xy'
    # arr_list = []
    # with open(coo_file, 'r') as f:
    #     degree = int(next(f))
    #     lines = f.readlines()
    #     for line in lines:
    #         a1, a2 = line.split(' ')
    #         arr_list.append([a1, a2])
    # arr = np.array(arr_list, dtype=float)
    # x_min = arr[:, 0].min()
    # x_max = arr[:, 0].max()
    # y_min = arr[:, 1].min()
    # y_max = arr[:, 1].max()
    # arr[:, 0] = (arr[:, 0] - x_min + 1e-5) / (x_max - x_min + 1e-4)
    # arr[:, 1] = (arr[:, 1] - y_min + 1e-5) / (y_max - y_min + 1e-4)
    lg, tm = main_flow(degree, arr, model, device, direction)
    sum_length0.append(lg)
    all_time.append(tm)
    # test_path = '/home/hrm/rsmt/algorithms/baseline/flute-3.1/test/'
    # degree = 40
    # file_name_list = os.listdir(test_path + str(degree))
    # for sample in tqdm(file_name_list):
    #     arr = np.loadtxt(test_path + str(degree) + '/' + sample)
    #     lg, tm = main_flow(degree, arr, direction=0)
    #     sum_length0.append(lg)
    #     all_time.append(tm)
        # error.append(er)
        # main_flow(degree, sample, direction=1)
        # main_flow(degree, sample, direction=2)
        # main_flow(degree, sample, direction=3)
        # main_flow(degree, sample, direction=4)
    print('平均长度: ', sum(sum_length0) / len(sum_length0))
    print('平均时间: ', sum(all_time) / len(all_time))
    print("长度方差：", np.var(sum_length0))
    print("长度标准差：", np.std(sum_length0))
    print()
    np.save('degree{}_length.npy'.format(degree), sum_length0)
    np.save('degree{}_time.npy'.format(degree), all_time)
    # print('平均ERROR: ', sum(error) / len(error))
    # opt_length = get_length(arr)
    # print("error:", (new_length / opt_length - 1) * 100, "%")