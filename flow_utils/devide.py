# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/9 18:44
# @Author      : gwsun
# @Project     : RSMT
# @File        : devide.py
# @env         : PyCharm
# @Description :
from matplotlib import pyplot as plt
import numpy as np
import time
from flow_utils.modui import generate_RMST3
from shared_so import generate_blocks as generate_blocks_cpp


def read_files(filename='n1944'):
    arr_list = []
    # degree = 1944
    coo_file = './'+filename+'.xy'

    with open(coo_file, 'r') as f:
        degree = int(next(f))
        lines = f.readlines()
        for line in lines:
            a1, a2 = line.split(' ')
            arr_list.append([a1, a2])
    arr = np.array(arr_list, dtype=float)
    x_min = min(arr[:, 0])
    y_min = min(arr[:, 1])
    arr[:, 0] = arr[:, 0] - x_min
    arr[:, 1] = arr[:, 1] - y_min
    plt.figure(figsize=(10, 10))
    x = arr[:, 0]
    y = arr[:, 1]
    plt.title(filename)
    plt.scatter(arr[:, 0], arr[:, 1], s=4)
    plt.savefig('./originScatter_' + filename + '.png')
    return arr


def post_process(mp, blocks, belongs, m_block=20, block_max=30):
    # 不会有两个小于20的block直接相连，所以只合并小于10的block至20的block
    # print('post processing(merge some small blocks together)')
    merge_into = np.array(list(range(len(blocks))))  # 初始时每个块的连接是他本身的id号

    link_block = np.zeros((len(blocks), len(blocks)), dtype=int)
    com20_id = []
    belongs = np.array(belongs)
    for i, block in enumerate(blocks):
        if len(block) == m_block:
            com20_id.append(i)
        for point in block:
            link_id = belongs[np.where(mp[point] == 1)[0]]
            link_block[i, link_id] = 1
            link_block[link_id, i] = 1
    # 删除本身的连接
    for i in range(len(blocks)):
        link_block[i, i] = 0

    # 按连接不完整块的多少升序
    com20_id.sort(key=lambda x: len([y for y in list(link_block[x]) if len(blocks[y]) < m_block]), reverse=False)
    blank_block = []
    for id in com20_id:  # 对正常块进行遍历
        linked = np.where(link_block[id] == 1)[0].tolist()  # np.where返回的是tuple（array，）
        linked.sort(key=lambda x: len(blocks[x]), reverse=False)  # 将其连接的block按从小到大排序
        block1 = blocks[id]
        for link_id in linked:
            block2 = blocks[link_id]
            if len(block2) != 0 and len(block1) + len(block2) <= block_max:
                blocks[id].extend(blocks[link_id])
                belongs[block2] = id
                merge_into[link_id] = id
                blocks[link_id].clear()
                blank_block.append(link_id)

            elif len(block1) + len(block2) > block_max:
                break
    blank_block.sort()
    for i in range(len(belongs)):
        index = len(np.where(blank_block <= belongs[i])[0])
        belongs[i] -= index
    new_blocks = [block for block in blocks if len(block) != 0]
    return new_blocks, belongs


def generate_blocks_2(arr, m_block=20, block_max=30, direction=0, detail=False):
    # print("deviding...")
    degree = arr.shape[0]
    # edges = generate_RMST(degree, arr)
    time1 = time.time()
    edges = generate_RMST3(degree, arr)
    time2 = time.time()
    # edges 转化为邻接矩阵形式
    mp = np.zeros([degree, degree], dtype=int)
    table = [list() for _ in range(degree)]
    for edge in edges:
        mp[edge[0]][edge[1]] = 1
        mp[edge[1]][edge[0]] = 1
        table[edge[0]].append(edge[1])
        table[edge[1]].append(edge[0])
    list(map(lambda x: x.sort(), table))
    blocks_20, belongs = generate_blocks_cpp.generate_blocks(arr.tolist(), table, m_block, block_max, direction=direction)
    time3 = time.time()
    final_blocks, belongs = post_process(mp, blocks_20, belongs, m_block=m_block, block_max=block_max)
    time4 = time.time()

    if detail:
        print("最终生成块数:{}".format(len(final_blocks)))
        print("块中最大数目:{}，块中最小数目:{}".format(max([len(bb) for bb in final_blocks]), min([len(bb) for bb in final_blocks])))
        print('The time of RMST generating:{}s. The time of BFS to generate blocks:{}s. '
              'The time of post process:{}s'.format(time2 - time1, time3 - time2, time4 - time3))
    return final_blocks, belongs, edges


if __name__ == '__main__':
    arr = np.random.rand(2000, 2)
    blocks_20, _, _ = generate_blocks_2(arr, detail=True, direction=0)

    print("block size:", [len(b) for b in blocks_20])
    print("all size:", sum([len(b) for b in blocks_20]))
    print()

