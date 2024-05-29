import numpy as np
import torch
import os
import time
import math
from models.gnn_merge import Actor, Critic
from utils.rsmt_utils import Evaluator
from utils.log_utils import *
import argparse
from torch.utils.tensorboard import SummaryWriter
from data.dataset import RandomDataset
from torch.utils.data import Dataset, DataLoader
from utils.myutil import eval_len_from_adj
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='NEW', help='experiment name')
parser.add_argument('--degree', type=int, default=40, help='maximum degree of nets')
parser.add_argument('--batch_size', type=int, default=128, help='test batch size')
parser.add_argument('--eval_size', type=int, default=10000, help='eval set size')
parser.add_argument('--total_sample', type=int, default=100000, help='total number of the sample')
parser.add_argument('--seed', type=int, default=9, help='random seed')
parser.add_argument('--gpu_id', type=str, default='0,1')
parser.add_argument('--learning_rate', type=float, default=0.00016)

parser.add_argument("--sync_bn", default=-1)
# parser.add_argument('--decay_rate', type=float, default=0.96)
# parser.add_argument('--decay_iter', type=int, default=5000)


# Hardcoded
log_intvl = 100

args = parser.parse_args()
# local_rank = args.local_rank
# torch.cuda.set_device(local_rank)


dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
device_id = local_rank % torch.cuda.device_count()

# device = torch.device("cuda:1")
# device = torch.device("cpu")

start_time = time.time()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print('experiment', args.experiment)
base_dir = 'save/'
exp_dir = base_dir + args.experiment + '/'
log_dir = exp_dir + 'rsmt' + str(args.degree) + '.log'
ckp_dir = exp_dir + 'rsmt' + str(args.degree) + '.pt'
best_ckp_dir = exp_dir + 'rsmt' + str(args.degree) + 'b.pt'
pre_ckp_dir = 'save/degree40_GCN_DDP_relu_glimpse/rsmt40b.pt'
tensor_root = './tensorboard4/'
cur_dir = '_lr_0.0002_degree'
if local_rank == 0:
    writer = SummaryWriter(tensor_root + args.experiment + cur_dir + str(args.degree))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print('Created exp_dir', exp_dir)
    else:
        print('Exp_dir', exp_dir, 'already exists')
    loger = LogIt(log_dir)
print('GPU{}已启动'.format(local_rank))
best_eval = 10.
best_kept = 0

actor = Actor().to(device_id)
critic = Critic().to(device_id)
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)
checkpoint = torch.load(pre_ckp_dir)
actor.load_state_dict(checkpoint['actor_state_dict'])
critic.load_state_dict(checkpoint['critic_state_dict'])
start_batch = 1

actor = DDP(actor, device_ids=[device_id])
critic = DDP(critic, device_ids=[device_id])

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.95)  # 学习率将在每100个周期乘以0.99
evaluator = Evaluator()

np.random.seed(args.seed + dist.get_rank())
torch.manual_seed(args.seed + dist.get_rank())

train_dataset = RandomDataset(args.total_sample, args.degree, file='./datamini')
eval_dataset = RandomDataset(args.eval_size, args.degree, file='./datamini')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=eval_sampler)

batch_id = 0
for epoch in range(100):
    train_loader.sampler.set_epoch(epoch+1)
    eval_loader.sampler.set_epoch(epoch+1)
    for data_sample in train_loader:
        batch_id += 1
        if local_rank == 0:
            print('\r目前batch:', batch_id, end='')
        actor.train()
        critic.train()
        arrs, adjs, adj_ins, adj_outs, masks, opt_len = data_sample
        arrs = arrs.to(device_id)
        adjs = adjs.to(device_id)
        adj_ins = adj_ins.to(device_id)
        adj_outs = adj_outs.to(device_id)
        masks = masks.to(device_id)

        new_adj, log_probs = actor(arrs, adj=adjs, mask_check=masks, adj_in=adj_ins, adj_out=adj_outs)
        # print('actor batch over!!!!!!!!!!', local_rank)
        predictions = critic(arrs, adjs, adj_ins, adj_outs)
        # print('critic batch over!!!!!!!!!!', local_rank)
        # 此时的output是包含点（0，0）的，还需要与之前的结果合并
        lengths = eval_len_from_adj(arrs, args.degree, new_adj)
        length_tensor = torch.tensor(lengths, dtype=torch.float).to(device_id)
        with torch.no_grad():
            disadvantage = length_tensor - predictions
        # print('disadvantage!!!!!!!!!!', local_rank)
        actor_loss = torch.mean(disadvantage * log_probs)
        critic_loss = mse_loss(predictions, length_tensor)
        loss = actor_loss + critic_loss
        # print('GPU{}的loss已回传'.format(local_rank))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        optimizer.step()
        # scheduler.step()
        # print('梯度已更新')
        if local_rank == 0:
            writer.add_scalar('train/actor_loss', actor_loss, batch_id)
            writer.add_scalar('train/critic_loss', critic_loss, batch_id)
            writer.add_scalar('train/loss', loss, batch_id)

        if batch_id % log_intvl == 0:
            if local_rank == 0:
                print('[batch', str(batch_id) + ',', 'time', str(int(time.time() - start_time)) + 's]')
                print('length ', lengths.mean())
                print('predict', predictions.cpu().detach().numpy().mean())
            actor.eval()
            eval_lengths = []
            error_batch = []
            for eval_batch in eval_loader:
                arrs, adjs, adj_ins, adj_outs, masks, opt_len = eval_batch
                arrs = arrs.to(device_id)
                adjs = adjs.to(device_id)
                adj_ins = adj_ins.to(device_id)
                adj_outs = adj_outs.to(device_id)
                masks = masks.to(device_id)
                opt_len = opt_len.tolist()
                with torch.no_grad():
                    new_adj, _ = actor(arrs, deterministic=True, adj=adjs, mask_check=masks, adj_in=adj_ins, adj_out=adj_outs)
                    lengths = eval_len_from_adj(arrs, args.degree, new_adj)
                eval_lengths.append(lengths)
                error_batch.append(round(((lengths / opt_len).mean() - 1) * 100, 3))

            error = torch.tensor(round(sum(error_batch) / len(error_batch), 3), device=device_id)
            # error = round(((np.concatenate(eval_lengths, -1) / gst_lengths).mean() - 1) * 100, 3)
            dist.all_reduce(error, op=dist.ReduceOp.AVG)
            eval_mean = np.concatenate(eval_lengths, -1).mean()
            eval_mean = torch.tensor(eval_mean, device=device_id)
            dist.all_reduce(eval_mean, op=dist.ReduceOp.AVG)
            if local_rank == 0:
                print('GNN method\'s percentage error  ', '{}%'.format(error))

                # eval_mean = np.concatenate(eval_lengths, -1).mean()
                if eval_mean < best_eval:
                    best_eval = eval_mean
                    best_kept = 0
                    # keep a checkpoint anyway
                    torch.save({
                        'batch_idx': batch_id,
                        'best_eval': best_eval,
                        'actor_state_dict': actor.module.state_dict(),
                        'critic_state_dict': critic.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, best_ckp_dir)
                    print('ckpt saved at', best_ckp_dir)
                else:
                    best_kept += 1
                    # if best_kept > 150:
                #         early stopping.
                print('eval', eval_mean.item())
                print('best', best_eval.item(), '(' + str(best_kept) + ')')
                print()
                loger.log_iter(batch_id, {'eval': eval_mean, 'best': best_eval, 'error': error,
                                           'time': int(time.time() - start_time)})
                writer.add_scalar('eval/eval_length', eval_mean, batch_id)
                writer.add_scalar('eval/eval_error', error, batch_id)

if local_rank == 0:
    torch.save({
        'batch_idx': batch_id,
        'best_eval': best_eval,
        'actor_state_dict': actor.module.state_dict(),
        'critic_state_dict': critic.module.state_dict(),
        'optimizer_state_dict': optimizer.module.state_dict()
    }, ckp_dir)
    print('ckpt saved at', ckp_dir)
    #
    # plot_curve(log_dir)
    print('process is over normaly')
