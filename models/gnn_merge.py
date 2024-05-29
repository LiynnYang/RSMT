# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/10/31 18:53
# @Author      : gwsun
# @Project     : RSMT
# @File        : gnn_merge.py
# @env         : PyCharm
# @Description :


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions.categorical import Categorical
from models.utils import Embedder, Pointer, Glimpse
from models.self_attn import Encoder
from models.gcn import GCNLayerV2


class TripleGNN(nn.Module):
    # TODO:注意力机制不对称的
    def __init__(self, in_feats, hid_feats, out_feats, channels=5):
        super().__init__()
        self.conv1 = GCNLayerV2(in_feats, hid_feats, channels)
        self.conv2 = GCNLayerV2(hid_feats, out_feats, channels)
        # self.bn1 = nn.BatchNorm1d(hid_feats)  # 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作
        # self.bn2 = nn.BatchNorm1d(out_feats)

    def forward(self, adj, adj_in, adj_out, inputs):
        # 输入是节点的特征
        batch = adj.shape[0]
        degree = adj.shape[-1]

        h = self.conv1(inputs, adj, adj_in, adj_out)
        # h = self.bn1(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        h = F.relu(h)
        # h = F.tanh(h)
        h = self.conv2(h, adj, adj_in, adj_out)
        # h = self.bn2(h.reshape(batch*degree, -1)).reshape(batch, degree, -1)
        h = F.relu(h)
        return h


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Tanh(),
        )
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
        self.bn = nn.BatchNorm1d(self.graph_hidden_size)

    def forward(self, input):
        graph_emb = (self.node_gating(input) * self.node_to_graph(input)).sum(1)
        graph_emb = self.bn(graph_emb)
        return graph_emb


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.device = None
        # embedder args
        self.d_input = 2
        self.d_model = 128
        self.embedder = Embedder(self.d_input, self.d_model)

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.seq = nn.Sequential(nn.Linear(self.d_model, 2 * self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(2*self.d_model, self.d_model))
        # feedforward layer inner
        self.d_inner = 512
        self.d_unit = 256
        # self.pos_ffn = PositionwiseFeedForward(self.d_model, self.d_inner)
        # self.encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)

        # decoder args
        # self.ptr = Pointer(2 * self.d_model, self.d_model)
        self.ptr = Pointer(self.d_model, self.d_model)
        # TODO: 需要考虑用于初始化的GAT是否需要与用于更新的GAT共享一个权重？
        self.gat = TripleGNN(self.d_model, self.d_model, self.d_model)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.BatchNorm1d(self.d_model)
        )
        # self.graph_emb = GraphEmbed(self.d_model)
        self.glimpse = Glimpse(self.d_model, self.d_unit)  # 若使用这个就要降ptr第一个参数设为d_model
        # self.bn = nn.BatchNorm1d(self.d_model)

    def update_mask(self, mask, edge_x, edge_y):
        """

        :param mask: [batch, degree, degree]
        :param edge: [batch, 2]
        :return:
        """
        batch = mask.shape[0]
        degree = mask.shape[-1]
        vir_adj = 1 - mask
        vir_adj = vir_adj.index_put((torch.arange(batch, dtype=torch.int64, device=mask.device), edge_x, edge_y),
                                    torch.LongTensor([1]).to(mask.device))
        vir_adj = vir_adj.index_put((torch.arange(batch, dtype=torch.int64, device=mask.device), edge_y, edge_x),
                                    torch.LongTensor([1]).to(mask.device))
        vir_adj -= torch.eye(degree, dtype=torch.int64, device=mask.device).unsqueeze(0).repeat(batch, 1, 1)
        vir_adj_2 = torch.matmul(vir_adj.to(torch.float32), vir_adj.to(torch.float32))
        vir_adj_3 = torch.matmul(vir_adj_2, vir_adj.to(torch.float32))
        vir_adj = (vir_adj + vir_adj_2 + vir_adj_3).to(torch.int64)
        vir_adj = torch.where(vir_adj >= 1, 1, vir_adj)
        new_mask = 1 - vir_adj
        new_mask[:, torch.arange(degree), torch.arange(degree)] = 0
        return new_mask

    def forward(self, inputs: torch.tensor, adj=None, mask_check=None, adj_in=None, adj_out=None, deterministic=False, single=False, valid=False, pad_len=None):
        """

        :param inputs: numpy.ndarray [batch_size * degree * 2]
        :param deterministic:
        :return:
        """
        batch_size = inputs.shape[0]
        degree = inputs.shape[1]
        time_start = time.time()
        # 获取当前进程模型所在位置设备
        self.device = inputs.device
        if mask_check is None and adj is not None:
            print('若输入带边的图，请同步输入相应mask矩阵')
            exit()
        if adj is None:
            adj = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            adj_in = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            adj_out = torch.eye(degree, dtype=torch.int64, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        if mask_check is None:
            mask_check = torch.ones([batch_size, degree, degree], dtype=torch.int64, device=self.device)
            self_loop = torch.eye(degree, dtype=torch.int64)
            self_loop = self_loop.unsqueeze(0).repeat(batch_size, 1, 1)
            mask_check -= self_loop.to(self.device)  # 去掉自连情况

        time_init = float(time.time() - time_start)
        indexes, log_probs = [], []

        embedings = self.embedder(inputs)
        encodings = self.seq(embedings)
        time_emb, time_update_graph, time_update_mask = .0, .0, .0
        # id_node_start = torch.arange(batch_size, dtype=torch.int64, device=self.device) * degree
        graph_mask = None
        if pad_len is not None:
            graph_mask = torch.arange(degree, device=inputs.device).repeat(batch_size, 1)
            tmp = pad_len.reshape(batch_size, -1)
            graph_mask = graph_mask.ge(degree - tmp)
        step = None
        for step in range(degree - 1):
            if torch.all(mask_check == 0):
                break
            time_start2 = time.time()
            node_embedding = self.gat(adj, adj_in, adj_out, encodings)
            input1 = node_embedding.repeat(1, 1, degree).reshape(batch_size, degree * degree, -1)
            input2 = node_embedding.repeat(1, degree, 1)
            final_input = torch.cat([input1, input2], dim=-1).reshape(batch_size * degree * degree, -1)
            edge_embedding = self.fc(final_input).reshape(batch_size, degree * degree, -1)
            node_embedding = node_embedding.reshape(batch_size, degree, -1)
            graph_embedding = self.glimpse(node_embedding, graph_mask)
            logits, t = self.ptr(edge_embedding, graph_embedding, mask_check)
            time_emb += float(time.time() - time_start2)
            distr = Categorical(logits=logits)
            if deterministic:
                _, edge_idx = torch.max(logits, -1)
            else:
                edge_idx = distr.sample()
            time_start3 = time.time()
            with torch.no_grad():
                x_idx = torch.div(edge_idx, degree, rounding_mode='floor')
                y_idx = torch.fmod(edge_idx, degree)
                # 在更新前先将mask pad部分全置为1，代表此点尚孤立; 更新后即复位为0，代表此点相关边不可选
                if pad_len is not None:
                    for id, pd in enumerate(pad_len):
                        if pd != 0:
                            mask_check[id][-pd:, :] = 1
                            mask_check[id][:, -pd:] = 1
                mask_check = self.update_mask(mask_check, x_idx, y_idx)
                if pad_len is not None:
                    for id, pd in enumerate(pad_len):
                        if pd != 0:
                            mask_check[id][-pd:, :] = 0
                            mask_check[id][:, -pd:] = 0
            time_update_mask += float(time.time() - time_start3)
            time_start4 = time.time()
            # 若某个图的连边已经结束，则让其默认取边（0，0）
            indexes.append(x_idx)
            indexes.append(y_idx)
            log_p = distr.log_prob(edge_idx)
            log_p[t] = log_p[t].detach()
            log_probs.append(log_p)
            adj = adj.index_put((torch.arange(batch_size, dtype=torch.int64, device=adj.device), x_idx, y_idx),
                                torch.LongTensor([1]).to(adj.device))

            tmp = adj[torch.arange(batch_size), x_idx]
            tmp[torch.arange(batch_size), x_idx] = 0
            tmp = tmp.unsqueeze(1).repeat(1, degree, 1)
            tmp = tmp * tmp.transpose(2, 1)
            adj_in = adj_in | tmp

            tmp = adj.transpose(2, 1)[torch.arange(batch_size), y_idx]
            tmp[torch.arange(batch_size), y_idx] = 0
            tmp = tmp.unsqueeze(1).repeat(1, degree, 1)
            tmp = tmp * tmp.transpose(2, 1)
            adj_out = adj_out | tmp

            time_update_graph += float(time.time() - time_start4)
            if single:  # 如果是找环过程，直接运行一次就返回
                return x_idx, y_idx
        # log_probs  9*1024
        log_probs = sum(log_probs)
        time_end = float(time.time() - time_start)
        # 返回形式：index：x1, y1, x2, y2, ..., 0, 0  (0,0)是为了凑齐长度所补的边，在长度计算中应该无碍
        # TODO：需验证一个或多个0，0边是否对长度无影响
        if valid:
            return adj, log_probs, step+1
        return adj, log_probs


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        # embedder args
        self.d_input = 2
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder(self.d_input, self.d_model)
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        # self.gnn = TripleGNN(self.d_model, self.d_model, self.d_model)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs, adj=None, adj_in=None, adj_out=None, deterministic=False):
        critic_encode = self.crit_encoder(self.crit_embedder(inputs), None)
        # critic_encode = self.gnn(adj, adj_in, adj_out, critic_encode)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions


if __name__ == '__main__':
    batch_size, degree, device = 1024, 3, 'cuda:0'
    actor = Actor()
    node = np.random.rand(batch_size, degree, 2)
    indexes, log_probs = actor(node)
    print(indexes)
