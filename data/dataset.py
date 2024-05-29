# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/28 19:50
# @Author      : gwsun
# @Project     : RSMT
# @File        : dataset.py
# @env         : PyCharm
# @Description :
from torch.utils.data import Dataset, DataLoader
from data.get_merge_data import get_data
import torch


class RandomDataset(Dataset):
    # 需要更改三处地方以获取ori_len
    """
    此类获取的数据集不含有原长度，只适用于训练
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./data'):
        arr, adj, adj_in, adj_out, mask, opt_len = get_data(num, degree, m_block, block_max, file=file)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index]

    def __len__(self):
        return len(self.arr)


class RandomDataset2(Dataset):
    """
    此类获取数据集含有原长度以及两块的大小，用于测试
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./datamini'):
        arr, adj, adj_in, adj_out, mask, opt_len, ori_len, ori_size, link_b = get_data(num, degree, m_block, block_max, file=file, mode=2)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)
        self.ori_len = torch.from_numpy(ori_len).to(torch.float32)
        self.ori_size = torch.from_numpy(ori_size).to(torch.int64)
        self.link_b = torch.from_numpy(link_b).to(torch.int64)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index], self.ori_len[index], self.ori_size[index], self.link_b[index]

    def __len__(self):
        return len(self.arr)


class RandomDataset3(Dataset):
    """
    此类只含有原长度，不含有大小，适用于测试
    """
    def __init__(self, num, degree, m_block=20, block_max=30, file='./datamini'):
        arr, adj, adj_in, adj_out, mask, opt_len, ori_len = get_data(num, degree, m_block, block_max, file=file, mode=2)
        # self.arr, self.adj, self.adj_in, self.adj_out, self.mask = get_data(num, degree)
        self.arr = torch.from_numpy(arr).to(torch.float32)
        self.adj = torch.from_numpy(adj).to(torch.int64)
        self.adj_in = torch.from_numpy(adj_in).to(torch.int64)
        self.adj_out = torch.from_numpy(adj_out).to(torch.int64)
        self.mask = torch.from_numpy(mask).to(torch.int64)
        self.opt_len = torch.from_numpy(opt_len).to(torch.float32)
        self.ori_len = torch.from_numpy(ori_len).to(torch.float32)

    def __getitem__(self, index):
        # return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index]
        return self.arr[index], self.adj[index], self.adj_in[index], self.adj_out[index], self.mask[index], \
               self.opt_len[index], self.ori_len[index]

    def __len__(self):
        return len(self.arr)


if __name__ == '__main__':
    degree = 40
    dataset = RandomDataset(100000, 40)
    dataloador = DataLoader(dataset, batch_size=256)
    for sample in dataloador:
        print(sample)
        break
