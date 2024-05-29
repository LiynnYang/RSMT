# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/11/10 19:13
# @Author      : gwsun
# @Project     : RSMT
# @File        : merge_test.py
# @env         : PyCharm
# @Description :
from models.gnn_merge import Actor
import torch
import argparse
from data.dataset import RandomDataset2, RandomDataset3
from utils.myutil import eval_len_from_adj, plot_merge_tune
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=40, help='maximum degree of nets')
parser.add_argument('--batch_size', type=int, default=128, help='test batch size')
parser.add_argument('--eval_size', type=int, default=10000, help='eval set size')

args = parser.parse_args()


best_ckpt = 'save/degree40_GCN_DDP_relu_glimpse/rsmt40b.pt'
device = "cuda:0"
checkpoint = torch.load(best_ckpt)
actor = Actor().to(device)
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()

eval_dataset = RandomDataset3(args.eval_size, args.degree, m_block=20, block_max=30, file='data/datatestDirect')
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

eval_lengths = []
error_batch = []
error_ori_batch = []
count_up = 0
for eval_batch in tqdm(eval_loader):
    # arrs, adjs, adj_ins, adj_outs, masks, opt_len, ori_len, ori_size, link_b = eval_batch
    arrs, adjs, adj_ins, adj_outs, masks, opt_len, ori_len = eval_batch
    arrs = arrs.to(device)
    adjs = adjs.to(device)
    adj_ins = adj_ins.to(device)
    adj_outs = adj_outs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        new_adj, _ = actor(arrs, deterministic=True, adj=adjs, mask_check=masks, adj_in=adj_ins, adj_out=adj_outs)
        lengths = eval_len_from_adj(arrs, args.degree, new_adj)
    count_up += (lengths < ori_len.numpy()).sum()
    lengths = np.where(lengths < ori_len.numpy(), lengths, ori_len.numpy())

    eval_lengths.append(lengths)
    error_batch.append(round(((lengths / opt_len).mean().item() - 1) * 100, 3))
    error_ori = round(((ori_len / opt_len).mean().item() - 1) * 100, 3)
    error_ori_batch.append(error_ori)


error = round(sum(error_batch) / len(error_batch), 3)
error_ori = round(sum(error_ori_batch) / len(error_ori_batch), 3)
print('模型错误率：', error)
print('原始devide错误率：', error_ori)
print('提升的数量：', count_up)
