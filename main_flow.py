# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/11/5 11:54
# @Author      : gwsun
# @Project     : RSMT
# @File        : main.py
# @env         : PyCharm
# @Description :
import copy
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.rsmt_utils import Evaluator, plot_gst_rsmt
from data.data_gen import get_optimal_graph
from flow_utils.devide import *
from flow_utils.findCircus import direction
from flow_utils.get_sqe import get_seqence, process_pair_data
import time
from utils.myutil import eval_len_from_adj, get_length, eval_adj_unbanlance


def get_graph_parallel(k, arr):
    """
    并行，产生邻接入矩阵与邻接出矩阵
    :param k:
    :param arr:
    :return:
    """
    a, b, length = get_optimal_graph(arr, mode='3', getL=True)
    return k, a, b, length


def get_level_data(adj_in1, adj_in2, adj_out1, adj_out2, arr1, arr2, l1, l2, between_blocks, belongs, pair, blocks_20, level=1):
    """

    :param adj_in1:
    :param adj_in2:
    :param adj_out1:
    :param adj_out2:
    :param arr1:
    :param arr2:
    :param level: merge层数，大于2时不会运行drop实例
    :return:
    """
    link = []  # link是在新的点集中的序号， e是在旧的点集中的序号
    ori_length = l1 + l2
    for e in between_blocks:
        if belongs[e[0]] == pair[0] and belongs[e[1]] == pair[1]:
            p1 = blocks_20[pair[0]].index(e[0])
            p2 = blocks_20[pair[1]].index(e[1]) + len(blocks_20[pair[0]])
            # link, add_len = direction(p1, p2, )
            link.append(p1)
            link.append(p2)
            # ori_length += add_len
            break
        elif belongs[e[0]] == pair[1] and belongs[e[1]] == pair[0]:
            p1 = blocks_20[pair[0]].index(e[1])
            p2 = blocks_20[pair[1]].index(e[0]) + len(blocks_20[pair[0]])
            link.append(p1)
            link.append(p2)
            # ori_length += (abs(arr[e[0]][0] - arr[e[1]][0]) + abs(arr[e[0]][1] - arr[e[1]][1]))
            break
    drop_data = None
    if level < 2:
        drop_data = process_pair_data(adj_in1, adj_in2, adj_out1, adj_out2, arr1, arr2, link)

    for k in range(len(adj_in2)):
        for q in range(len(adj_in2[k])):
            adj_in2[k][q] += len(arr1)
        for q in range(len(adj_out2[k])):
            adj_out2[k][q] += len(arr1)
    a_in, a_out = [], []
    a_in.extend(adj_in1)
    a_out.extend(adj_out1)
    a_in.extend(adj_in2)
    a_out.extend(adj_out2)

    add_edge, add_length = direction(link[0], link[1], a_in, a_out, np.concatenate((arr1, arr2), axis=0))
    a_in[add_edge[1]].append(add_edge[0])  # 默认link0指向link1
    a_out[add_edge[0]].append(add_edge[1])
    ori_length += add_length
    if level < 2:
        return drop_data, (a_in, a_out, ori_length), len(arr1) + len(arr2)
    return (a_in, a_out, ori_length), len(arr1) + len(arr2)


def main_flow(degree, arr, model, device, direction=0):
    pool1 = mp.Pool(mp.cpu_count())
    all_time = time.time()
    start_time = time.time()
    blocks_20, belongs, edges = generate_blocks_2(arr, direction=direction, detail=False)
    belongs = np.array(belongs)
    blocks_gen = time.time()

    adj_in, adj_out, length = [], [], []  # 共享数据

    def collect_result(result):
        k, a, b, lt = result
        adj_in.append((k, a))
        adj_out.append((k, b))
        length.append((k, lt))

    time2 = time.time()
    # eva = Evaluator()
    for i, block in enumerate(blocks_20):
        # a, b, lt = get_optimal_graph(arr[block], mode='3', getL=True)
        # gst_length, sp_list, edge_list = eva.gst_rsmt(arr[block])
        # plot_gst_rsmt(arr[block], sp_list, edge_list)

        sub_points = arr[block]
        pool1.apply_async(get_graph_parallel, args=(i, sub_points), callback=collect_result)

        # adj_in.append(a)
        # adj_out.append(b)
        # length.append(lt)
    # 将结果进行排序
    # 关闭进程、回收进程
    pool1.close()
    pool1.join()
    adj_in.sort(key=lambda x: x[0])
    adj_out.sort(key=lambda x: x[0])
    length.sort(key=lambda x: x[0])
    adj_in = [r[1] for r in adj_in]
    adj_out = [r[1] for r in adj_out]
    length = [r[1] for r in length]

    num_block = len(blocks_20)
    graph_gen = time.time()


    block_link = np.zeros([num_block, num_block], dtype=int)
    between_blocks = []
    for edge in edges:
        u, v = edge[0], edge[1]
        if belongs[u] != belongs[v]:
            block_link[belongs[u], belongs[v]] = 1
            block_link[belongs[v], belongs[u]] = 1
            between_blocks.append([u, v])
            # plt.plot([arr[u][0], arr[u][0]], [[arr[u][1], arr[v][1]]], color='r')
    #         plt.plot([arr[u][0], arr[u][0]], [arr[u][1], arr[v][1]], color='r')
    #         plt.plot([arr[u][0], arr[v][0]], [arr[v][1], arr[v][1]], color='r')
    # plt.savefig('opt_rsmt.png')
    # 2. 确定两两merge的顺序
    # print('Start to get merge sequence.')
    block_size = [len(b) for b in blocks_20]
    block_size = np.array(block_size)
    ans, records = get_seqence(block_link, block_size)

    # def collect_data()
    # 3. 构造送进模型的数据, 每次merge改变的只有block中的包含点序号以及相应块得邻接表，总arr不变
    # 为了节省时间，只对前两层merge使用模型
    level = 0
    time_data = 0
    time_model = 0
    eval_time = 0
    time_merge = time.time()

    for level_merge, record in zip(ans, records):
        level += 1
        # pool2 = mp.Pool(len(level_merge))
        # TODO:数据并行处理
        new_arr, new_adj, new_second_in, new_second_out, new_mask, real_de = [], [], [], [], [], []
        cat_adj_in, cat_adj_out, ori_length = [], [], []
        ori_pair_size = []
        max_degree = 0
        num_merge = len(level_merge)  # merge的块对数
        time_data_start = time.time()
        for pair, r in zip(level_merge, record):
            l1, l2 = length[pair[0]], length[pair[1]]
            if level < 2:
                ori_pair_size.append([len(blocks_20[pair[0]]), len(blocks_20[pair[1]])])
                model_data, cat_data, de = get_level_data(adj_in[pair[0]], adj_in[pair[1]], adj_out[pair[0]],
                                                          adj_out[pair[1]], arr[blocks_20[pair[0]]],
                                                          arr[blocks_20[pair[1]]], l1, l2, between_blocks, belongs,
                                                          pair, blocks_20, level)
                max_degree = max(max_degree, de)
                real_de.append(de)
                new_arr.append(model_data[0])
                new_adj.append(model_data[1])
                new_second_in.append(model_data[2])
                new_second_out.append(model_data[3])
                new_mask.append(model_data[4])
                cat_adj_in.append(cat_data[0])
                cat_adj_out.append(cat_data[1])
                ori_length.append(cat_data[2])
            else:
                cat_data, de = get_level_data(adj_in[pair[0]], adj_in[pair[1]], adj_out[pair[0]],
                                                          adj_out[pair[1]], arr[blocks_20[pair[0]]],
                                                          arr[blocks_20[pair[1]]], l1, l2, between_blocks, belongs,
                                                          pair, blocks_20, level)
                cat_adj_in.append(cat_data[0])
                cat_adj_out.append(cat_data[1])
                ori_length.append(cat_data[2])

        if level < 2:
            # 补全数据至最大degree
            pad_len = []
            orig_arr = []
            for i in range(num_merge):
                need = max_degree - real_de[i]
                pad_len.append(need)
                ori_arr = copy.deepcopy(new_arr[i])
                orig_arr.append(F.pad(ori_arr, [0, 0, 0, need]))  # 原坐标数据

                x_min = new_arr[i][0, :, 0].min()
                x_max = new_arr[i][0, :, 0].max()
                y_min = new_arr[i][0, :, 1].min()
                y_max = new_arr[i][0, :, 1].max()
                new_arr[i][0, :, 0] = (new_arr[i][0, :, 0] - x_min + 1e-5) / (x_max - x_min + 1e-4)
                new_arr[i][0, :, 1] = (new_arr[i][0, :, 1] - y_min + 1e-5) / (y_max - y_min + 1e-4)

                new_arr[i] = F.pad(new_arr[i], [0, 0, 0, need])
                new_adj[i] = F.pad(new_adj[i], [0, need, 0, need])
                new_second_in[i] = F.pad(new_second_in[i], [0, need, 0, need])
                new_second_out[i] = F.pad(new_second_out[i], [0, need, 0, need])
                new_mask[i] = F.pad(new_mask[i], [0, need, 0, need])
            eval_arr = torch.cat(orig_arr)
            final_arr = torch.cat(new_arr)
            final_adj = torch.cat(new_adj)
            final_second_in = torch.cat(new_second_in)
            final_second_out = torch.cat(new_second_out)
            final_mask = torch.cat(new_mask)
            pad_len = torch.LongTensor(pad_len)
            time_data_end = time.time()
            time_data += (time_data_end - time_data_start)
            time_model_start = time.time()
            adj, _, step = model(final_arr.to(device), final_adj.to(device), final_mask.to(device),
                                 final_second_in.to(device), final_second_out.to(device), pad_len=pad_len.to(device),
                                 deterministic=True,
                                 single=False, valid=True)
            time_model += (time.time() - time_model_start)
            # 计算模型产生的邻接矩阵的长度, 评价后就已经减去对角线上的1
            eval_time_start = time.time()
            # new_length = eval_adj_unbanlance(adj, final_arr, real_de)
            new_length = eval_adj_unbanlance(adj, eval_arr, real_de)
            eval_time += (time.time() - eval_time_start)
            # 一一比较并做出处理
            # ft_adj_in, ft_adj_out = [], []  # 用于微调的邻接表
            # ft_length = []  # 用于微调的长度列表
            for i, pair in enumerate(level_merge):
                blocks_20[pair[0]].extend(blocks_20[pair[1]])
                assert adj[i].reshape(-1).sum() == real_de[i] - 1
                if new_length[i] < ori_length[i]:
                    # print("model optimzation!")
                    # ft_length.append(new_length[i])
                    a_in, a_out = [], []
                    # 将当前adj裁剪到真实大小
                    real_adj = adj[i][:real_de[i], :real_de[i]]
                    for row in real_adj:
                        a_out.append(torch.where(row == 1)[0].tolist())
                    for row in real_adj.transpose(0, 1):
                        a_in.append(torch.where(row == 1)[0].tolist())
                    # a_in, a_out = finetune(10, a_in, a_out, real_adj, arr[blocks_20[pair[0]]])
                    adj_in[pair[0]] = a_in
                    adj_out[pair[0]] = a_out
                    length[pair[0]] = new_length[i]

                else:
                    # ft_length.append(ori_length[i])
                    # real_adj = torch.eye(len(cat_adj_out[i]), dtype=torch.int64)
                    # for p, row in enumerate(cat_adj_out[i]):
                    #     real_adj[p, row] = 1
                    # a_in, a_out = finetune(10, cat_adj_in[i], cat_adj_out[i], real_adj, arr[blocks_20[pair[0]]])
                    adj_in[pair[0]] = cat_adj_in[i]
                    adj_out[pair[0]] = cat_adj_out[i]
                    length[pair[0]] = ori_length[i]
                # ft_adj_in.append(adj_in[pair[0]])
                # ft_adj_out.append(adj_out[pair[0]])
            #  FineTune on the first two layers
            # print("第{}层merge finetune".format(level))
            # all_in, all_out = fineTune(10, ft_adj_in, ft_adj_out, max_degree, model, pad_len, final_arr, ft_length, ori_pair_size)
            # for i, pair in enumerate(level_merge):
            #     adj_in[pair[0]] = all_in[i]
            #     adj_out[pair[0]] = all_out[i]
            #     length[pair[0]] = ft_length[i]
            # assert len(adj_in[pair[0]]) == len(blocks_20[pair[0]])
            # for i, a in enumerate(adj_in[pair[0]]):
            #     for p in a:
            #         assert i in adj_out[pair[0]][p], "error"

        else:
            for i, pair in enumerate(level_merge):
                blocks_20[pair[0]].extend(blocks_20[pair[1]])
                adj_in[pair[0]] = cat_adj_in[i]
                adj_out[pair[0]] = cat_adj_out[i]
                length[pair[0]] = ori_length[i]
        # pool2.close()
        # pool2.join()
        # 执行完一层的merge，开始调整结构，adj_in, adj_out, blocks_20, belongs
        record.sort(reverse=True)
        for r in record:
            # 删除r
            adj_in.pop(r)
            adj_out.pop(r)
            length.pop(r)
            blocks_20.pop(r)
        for i, b in enumerate(blocks_20):
            for p in b:
                belongs[p] = i

    all_time_cost = time.time() - all_time
    # print('The time of optimal graphs generation is:', graph_gen - time2, 's')
    # print('The time of blocks generation is:', blocks_gen - start_time, 's')
    # print("merge时间耗费：", time.time() - time_merge, 's')
    # print("其中数据处理merge耗费", time_data, 's')
    # print("其中model-time cost:", time_model)
    # print("其中评价长度耗费：", eval_time)
    # print("all time cost:", time.time() - all_time)
    # plt.savefig('opt_rsmt2.png')
    adj = np.eye(degree, dtype=int)  # 自环的邻接矩阵
    for row, out in enumerate(adj_out[0]):
        adj[row, out] = 1
    assert adj.reshape(-1).sum() == degree * 2 - 1, 'merge error!'
    new_length = eval_len_from_adj(arr[blocks_20[0]], degree, adj)[0]
    # print("length:", new_length)
    # opt_length = get_length(arr)
    # print("error:", (new_length / opt_length - 1) * 100, "%")
    return new_length, all_time_cost  # , (new_length / opt_length - 1) * 100
