# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/23 15:08
# @Author      : gwsun
# @Project     : RSMT
# @File        : get_merge_data.py
# @env         : PyCharm
# @Description :

import pickle
from flow_utils.devide import *
from utils.myutil import get_length
from data.data_gen import *
import os
import sys
from flow_utils.findCircus import direction

sys.path.append('/data/data/rsmt')


def oriAlgo2(X0, X):
    min_last1 = 99999.9  # 最小的距离
    min_last1_index = -1
    min_last2 = 100000.0  # 倒数第二小的距离
    min_last2_index = -1
    for i in range(len(X)):
        tmp_dis = abs(X[i][0] - X0[0]) + abs(X[i][1] - X0[1])
        if tmp_dis < min_last1:
            min_last2 = min_last1
            min_last2_index = min_last1_index
            min_last1 = tmp_dis
            min_last1_index = i
        elif tmp_dis >= min_last1 and tmp_dis < min_last2:
            min_last2 = tmp_dis
            min_last2_index = i
        else:
            pass
    return [min_last1_index, min_last2_index]


# 邻接表实现的DFS
def dfs(block: list, visited: np.ndarray, start: int, adj_table: list):  # 从第i个点的第j个邻居开始, start=0, i=start, j=0
    if visited[start]:
        return
    block.append(start)
    visited[start] = 1
    for i in range(0, len(adj_table[start])):
        dfs(block, visited, adj_table[start][i], adj_table)


def get_data(num, degree, m_block=20, block_max=30, is_plot=False, file='./data', mode=1):
    if not os.path.exists(file):
        os.makedirs(file)
    file = file + '/merge_data_num{}_degree{}.pkl'.format(num, degree)
    maxC = 0
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    get_num = 0
    print('正在生成数据...')
    arrs, adjs, adj_ins, adj_outs, masks, opt_len, ori_lens, ori_size, link_b = [], [], [], [], [], [], [], [], []
    while 1:
        if get_num == num:
            break
        case = np.random.rand(degree, 2)
        blocks_20, belongs, edges = generate_blocks_2(case, m_block=m_block, block_max=block_max)
        if len(blocks_20) > 2:
            continue
        get_num += 1
        l = get_length(case)
        opt_len.append(l)
        ori_size.append([len(b) for b in blocks_20])
        n1, n2 = len(blocks_20[0]), len(blocks_20[1])
        new_case = np.concatenate((case[blocks_20[0]], case[blocks_20[1]]), axis=0)
        print('\r当前进度：{0}{1}%'.format('▉' * int(get_num / num * 60), round(get_num / num * 100, 2)), end='')
        if is_plot:
            fig = plt.figure(figsize=(10, 4.6))
            plt.subplot(1, 2, 1)
            for i, p in enumerate(new_case):
                plt.text(p[0]+0.01, p[1]+0.01, str(i))
        arrs.append(new_case)
        new_blocks = [list(range(n1)), list(range(n1, n1 + n2))]
        block_link = []
        ori_len = 0
        for edge in edges:
            u, v = edge[0], edge[1]
            if belongs[u] != belongs[v]:
                if belongs[u] == 0:
                    block_link.append(blocks_20[belongs[u]].index(u))
                    block_link.append(blocks_20[belongs[v]].index(v) + n1)
                else:  # belongs[v] == 0
                    block_link.append(blocks_20[belongs[u]].index(u) + n1)
                    block_link.append(blocks_20[belongs[v]].index(v))
                break

        block_center = []

        adj_in, adj_out = [], []

        for i, block in enumerate(new_blocks):
            sub_points = new_case[block]
            x_center = sub_points[:, 0].mean()
            y_center = sub_points[:, 1].mean()
            block_center.append([x_center, y_center])
            a_in, a_out, length = get_optimal_graph(sub_points, mode='3', getL=True)
            ori_len += length
            for k in range(len(a_in)):
                for q in range(len(a_in[k])):
                    a_in[k][q] += i * n1
                for q in range(len(a_out[k])):
                    a_out[k][q] += i * n1

            adj_in.extend(a_in)
            adj_out.extend(a_out)
            if is_plot:
                l, sp_list, edge_list = evaluator.gst_rsmt(sub_points)
                plot_gst_rsmt(sub_points, sp_list, edge_list, 'b', 'b')
        add_edge, add_len = direction(block_link[0], block_link[1], adj_in, adj_out, new_case)
        ori_lens.append(ori_len + add_len)
        # 添加块间的连接
        link_b.append(add_edge)
        adj_in[add_edge[1]].append(add_edge[0])
        adj_out[add_edge[0]].append(add_edge[1])
        # tmp_adj_in = copy.deepcopy(adj_in)
        # tmp_adj_out = copy.deepcopy(adj_out)
        block_center = np.array(block_center)
        if is_plot:
            plt.scatter(block_center[:, 0], block_center[:, 1], c='r', s=15)
            plt.plot(new_case[block_link, 0], new_case[block_link, 1], color='r')

        drop_point = []
        # 重心不对的点
        for i, point in enumerate(new_case):
            distance_all = []
            for center in block_center:
                distance_all.append((abs(center[0] - point[0]) + abs(center[1] - point[1])))
            distance_all = np.array(distance_all)
            if distance_all.argmin() != (i >= n1):
                drop_point.append(i)
                if is_plot:
                    plt.scatter(point[0]-0.01, point[1]-0.01, s=30, c='r', marker='^')
        drop_point = list(set(drop_point))

        final_dropPoint = []
        final_dropPoint.extend(drop_point)
        final_dropPoint.extend(block_link)
        # # 离重心最近的其他块的点
        # for point in drop_point:
        #     if point // n1 == 0:
        #         res = oriAlgo2(new_case[point], new_case[new_blocks[1]])
        #         res[0] += n1
        #         res[1] += n1
        #     else:
        #         res = oriAlgo2(new_case[point], new_case[new_blocks[0]])
        #     # plt.scatter(new_case[res, 0], new_case[res, 1], s=100, c='g', marker='*')
        #     final_dropPoint.extend(res)
        final_dropPoint = list(set(final_dropPoint))  # 在整张图中的序号

        numlist = []  # 用来找dfs的连通分量
        numlist.extend(final_dropPoint)
        # # 判断是否为共有一个steiner point，有的话则去掉对应边，对应mask置1
        count = 0
        ## 指出
        for i in final_dropPoint:
            for j in adj_out[i]:
                set1 = adj_in[j]
                numlist.append(j)
                for m in set1:
                    if (new_case[i, 0] < new_case[j, 0] and new_case[m, 0] < new_case[j, 0]) or \
                            (new_case[i, 0] > new_case[j, 0] and new_case[m, 0] > new_case[j, 0]):
                        adj_in[j].remove(m)
                        adj_out[m].remove(j)
                        numlist.append(m)
                        count += 1
                        if is_plot:
                            plt.scatter(new_case[m, 0], new_case[m, 1], s=50, c='y', marker='^')
                        # mask[m][j]=1

        ## 指入
        for i in final_dropPoint:
            for j in adj_in[i]:
                set2 = adj_out[j]
                numlist.append(j)
                for m in set2:
                    if (new_case[i, 1] < new_case[j, 1] and new_case[m, 1] < new_case[j, 1]) or \
                            (new_case[i, 1] > new_case[j, 1] and new_case[m, 1] > new_case[j, 1]):
                        adj_out[j].remove(m)
                        adj_in[m].remove(j)
                        numlist.append(m)
                        count += 1
                        if is_plot:
                            plt.scatter(new_case[m, 0], new_case[m, 1], s=50, c='y', marker='^')
                        # mask[j][m]=1

        # 去掉点之后的邻接表
        for i in final_dropPoint:
            numlist.extend(adj_in[i])
            numlist.extend(adj_out[i])
            count += len(adj_in[i])
            count += len(adj_out[i])
            adj_in[i] = []
            adj_out[i] = []
        for k in range(degree):
            for i in final_dropPoint:
                if i in adj_in[k]:
                    adj_in[k].remove(i)
                if i in adj_out[k]:
                    adj_out[k].remove(i)
        maxC = max(maxC, count)
        # 去掉点之后的邻接矩阵
        adj = np.eye(degree)
        for i in range(n1 + n2):
            adj[i, adj_out[i]] = 1

        list1 = list(set(numlist))  # 所有涉及到的点
        adjs.append(adj)
        if is_plot:
            plt.scatter(new_case[list1, 0], new_case[list1, 1], c='none', marker='o', edgecolors='r', s=100)

        adj_table = []
        for i in range(len(adj_in)):
            adj_table.append(adj_in[i] + adj_out[i])

        mask = np.ones([degree, degree], dtype=bool)
        row, col = np.diag_indices_from(mask)
        mask[row, col] = 0
        for start in list1:
            block = []
            visited = np.zeros(n1 + n2)
            dfs(block, visited, start, adj_table)
            # 对块内元素mask置为0
            tmp = np.ones([degree, degree], dtype=bool)
            tmp[block, :] = 0
            tmp = tmp | tmp.T
            mask = mask & tmp
        masks.append(mask)
        # 获取二阶入度、出度矩阵
        second_adj_in = np.eye(degree, dtype=bool)
        second_adj_out = np.eye(degree, dtype=bool)
        for i in range(degree):
            tmp = np.zeros([degree, degree], dtype=bool)
            tmp[adj_out[i], :] = 1
            tmp = tmp & tmp.T
            second_adj_in |= tmp

            tmp = np.zeros([degree, degree], dtype=bool)
            tmp[adj_in[i], :] = 1
            tmp = tmp & tmp.T
            second_adj_out |= tmp

        adj_ins.append(second_adj_in)
        adj_outs.append(second_adj_out)
        # output = np.stack(np.where(np.triu(adj) == 1)).T.reshape(-1)
        # plt.subplot(1, 2, 2)
        # plot_rest(new_case, output)
        if is_plot and count >= 30:
            fig.savefig('./benchmark/figure/contrast4.png')
            print()
        plt.cla()
    print('\n生成完成')
    print('degree{}图中最大删减边数量:'.format(degree), maxC)
    # 保存训练数据集
    if mode == 1:
        save_content = np.stack(arrs, 0), np.stack(adjs, 0), np.stack(adj_ins, 0), \
                       np.stack(adj_outs, 0), np.stack(masks, 0), np.array(opt_len)  # , np.array(ori_lens)
    else:
        save_content = np.stack(arrs, 0), np.stack(adjs, 0), np.stack(adj_ins, 0), \
                       np.stack(adj_outs, 0), np.stack(masks, 0), np.array(opt_len), \
                       np.array(ori_lens), np.array(ori_size), np.array(link_b)
    with open(file, 'wb') as f:
        pickle.dump(save_content, f)
    print('内容已成功保存至', file)
    return save_content


if __name__ == '__main__':
    # 产生数据，组织成arr, adj， adj_in, adj_out, mask的形式
    # 是否注释掉143-152行
    degree = 40
    batch_size = 100000
    # get_data(100000, 40)
    get_data(100000, 80, m_block=40, block_max=60)
    get_data(10000, 80, m_block=40, block_max=60)
    # get_data(100000, 20, 10, 15)
    # get_data(10000, 20, 10, 15)
