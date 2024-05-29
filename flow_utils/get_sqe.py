# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/11/23 19:47
# @Author      : gwsun
# @Project     : RSMT-main
# @File        : get_sqe.py
# @env         : PyCharm
# @Description :
import numpy as np
from flow_utils.findCircus import foundBreak, cengravity, direction
import copy
import torch


def get_seqence(block_link, block_size):
    """
    get the seqence of merge
    :param block_link:
    :return:
    """
    block_degree = block_link.sum(1)
    num_block = block_link.shape[0]
    priority = np.zeros([num_block])  # 每个块的merge层级, 如果被合并则置为inf
    cur_priority = 0  # 当前merge层级
    ans = []
    visited = np.zeros(num_block)
    records = []  # 记录被合并块
    record = []

    def cmp(x):
        return priority[x], abs(block_size[x]-20), block_degree[x]

    def cmp2(x):
        return abs(block_size[x]), block_degree[x]

    while num_block != 1:
        pre_level = np.where(priority < cur_priority)[0]
        un_visited = np.where(visited == 0)[0]
        pre_level = pre_level[np.in1d(pre_level, un_visited)]
        if len(pre_level) != 0:
            # TODO:先按pre_leval排一个升序，再按degree升序， 再按大小升序排序
            a = pre_level.tolist()
            a.sort(key=cmp)
            select1 = a[0]
        else:
            un_merge = np.where(priority == cur_priority)[0]
            un_merge = un_merge[np.in1d(un_merge, un_visited)]
            # 选择一个degree最小且大小最小的
            un_merge = un_merge.tolist()
            un_merge.sort(key=cmp2)
            select1 = un_merge[0]
        visited[select1] = 1
        # 再从所连接的块中选一个degree最小的block
        linked = np.where(block_link[select1] == 1)[0]
        un_visited = np.where(visited == 0)[0]
        linked = linked[np.in1d(linked, un_visited)]
        if len(linked) != 0:
            linked = linked.tolist()
            linked.sort(key=cmp2)
            # min_degree = block_degree[linked].min()
            # min_index = np.where(block_degree[linked] == min_degree)[0]
            # select2 = linked[min_index[block_size[linked][min_index].argmin()]]
            select2 = linked[0]
            visited[select2] = 1
            num_block -= 1
            if len(ans) <= cur_priority:
                ans.append([])
            ans[cur_priority].append([select1, select2])
            priority[select1] = cur_priority + 1
            # 处理select2, 将其孤立，并设置merge层级为无穷
            record.append(select2)
            block_link[select1, :] |= block_link[select2, :]
            block_link[:, select1] |= block_link[:, select2]
            block_link[select2, :] = 0
            block_link[:, select2] = 0
            block_link[select1, select1] = 0  # 避免块merge选到自身
            priority[select2] = float('inf')
            block_degree[select1] += block_degree[select2]
            block_degree[select1] -= 1  # 减去连接自身
            block_size[select1] += block_size[select2]
        if np.all(visited):
            # 所有的块都已经被访问
            cur_priority += 1
            visited = np.zeros(num_block)
            # TODO: block_degree, priority, block_link, block_size
            block_link = np.delete(block_link, record, axis=0)
            block_link = np.delete(block_link, record, axis=1)  # block_link本身不变化
            priority = np.delete(priority, record, axis=0)
            block_degree = block_link.sum(1)
            block_size = np.delete(block_size, record, 0)
            records.append(record)  # 记录被删除的block的id
            record = list()
    return ans, records


# 邻接表实现的DFS
def dfs(block: list, visited: np.ndarray, start: int, adj_table: list):  # 从第i个点的第j个邻居开始, start=0, i=start, j=0
    if visited[start]:
        return
    block.append(start)
    visited[start] = 1
    for i in range(0, len(adj_table[start])):
        dfs(block, visited, adj_table[start][i], adj_table)


def process_pair_data(adj_in1, adj_in2, adj_out1, adj_out2, arr1, arr2, link, detail=False):
    n1, n2 = len(arr1), len(arr2)
    new_case = np.concatenate((arr1, arr2), axis=0)
    new_blocks = [list(range(n1)), list(range(n1, n1 + n2))]
    block_center = []
    block_center.append([arr1[:, 0].mean(), arr1[:, 1].mean()])
    block_center.append([arr2[:, 0].mean(), arr2[:, 1].mean()])
    if n1 > n2:  # arr1是大块
        new_center = cengravity(block_center[0][0], block_center[0][1], block_center[1][0], block_center[1][1],
                                n1, n2)
        block_center[0] = new_center[0]
    else:                      # arr2是大块
        new_center = cengravity(block_center[1][0], block_center[1][1], block_center[0][0], block_center[0][1],
                                n2, n1)
        block_center[1] = new_center[0]
    degree = n1 + n2
    adj_in, adj_out = copy.deepcopy(adj_in1), copy.deepcopy(adj_out1)
    adj_in2, adj_out2 = copy.deepcopy(adj_in2), copy.deepcopy(adj_out2)
    for k in range(len(adj_in2)):
        for q in range(len(adj_in2[k])):
            adj_in2[k][q] += n1
        for q in range(len(adj_out2[k])):
            adj_out2[k][q] += n1
    adj_in.extend(adj_in2)
    adj_out.extend(adj_out2)

    drop_point = []
    # 重心不对的点
    for i, point in enumerate(new_case):
        distance_all = []
        for center in block_center:
            distance_all.append((abs(center[0] - point[0]) + abs(center[1] - point[1])))
        distance_all = np.array(distance_all)
        if distance_all.argmin() != (i >= n1):
            drop_point.append(i)

    drop_point = list(set(drop_point))

    final_dropPoint = []
    final_dropPoint.extend(drop_point)
    final_dropPoint.extend(link)

    final_dropPoint = list(set(final_dropPoint))  # 在整张图中的序号

    numlist = []  # 用来找dfs的连通分量
    numlist.extend(final_dropPoint)
    # # 判断是否为共有一个steiner point，有的话则去掉对应边
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
    if detail:
        print('n1:{}, n2:{}删除的边数：'.format(n1, n2), count)
    # 去掉点之后的邻接矩阵
    adj = np.eye(degree)
    for i in range(n1 + n2):
        adj[i, adj_out[i]] = 1

    list1 = list(set(numlist))  # 所有涉及到的点
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

    save_content = torch.from_numpy(new_case).to(torch.float32).unsqueeze(0), \
                   torch.from_numpy(adj).to(torch.int64).unsqueeze(0), \
                   torch.from_numpy(second_adj_in).to(torch.int64).unsqueeze(0), \
                   torch.from_numpy(second_adj_out).to(torch.int64).unsqueeze(0), \
                   torch.from_numpy(mask).to(torch.int64).unsqueeze(0)
    return save_content
