import argparse
import time
import math
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from HASNN import dataset, neuron
from HASNN.layers import SRGA
from HASNN.utils import (RandomWalkSampler, Sampler, add_selfloops,
                         set_seed, tab_printer)


def get_sinusoid_encoding_table(n_position, d_hid, device):
    ''' Sinusoidal position encoding table '''
    def get_angle(pos, i):
        return pos / (10000 ** (2 * (i // 2) / d_hid))
    table = torch.zeros(n_position, d_hid, device=device)
    for pos in range(n_position):
        for i in range(d_hid):
            angle = get_angle(pos, i)
            table[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return table


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, max_len=None):
        super().__init__()
        self.attn_fc = nn.Linear(input_dim, 1)
        self.max_len = max_len
        if max_len:
            pe = get_sinusoid_encoding_table(max_len, input_dim, device=torch.device('cuda:0'))
            self.register_buffer('pos_embedding', pe)

    def forward(self, seq):  # seq: [T, B, H]
        T, B, H = seq.shape
        if self.max_len:
            pos = self.pos_embedding[:T].unsqueeze(1).to(seq.device)  # [T,1,H]
            seq = seq + pos
        scores = self.attn_fc(seq.permute(1, 0, 2))  # [B,T,1]
        weights = torch.softmax(scores, dim=1)
        out = torch.sum(seq.permute(1, 0, 2) * weights, dim=1)  # [B,H]
        return out


class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):  # x1,x2: [B,H]
        Q = self.q(x1).unsqueeze(1)        # [B,1,H]
        K = self.k(x2).unsqueeze(2)        # [B,H,1]
        scores = torch.bmm(Q, K)           # [B,1,1]
        weights = self.softmax(scores)     # [B,1,1]
        V = self.v(x2).unsqueeze(1)        # [B,1,H]
        return (weights * V).squeeze(1)    # [B,H]


class HASNN(nn.Module):
    def __init__(self, in_features, out_features, hids=[512, 64], alpha=1.0, p=0.5,
                 dropout=0.7, bias=False, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF',
                 nchannels=2, invth=1, time_mod=2, w_pos=0.6, w_nopos=0.4):
        super().__init__()
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(add_selfloops(adj)) for adj in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(adj)) for adj in data.adj_evolve]
        else:
            self.sampler = [Sampler(add_selfloops(adj)) for adj in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj)) for adj in data.adj_evolve]
        self.SRGA = nn.ModuleList([
            SRGA(in_features, hids=hids, sizes=sizes, v_threshold=invth,
                    alpha=alpha, surrogate=surrogate, concat=concat,
                    bias=bias, aggr=aggr, dropout=dropout)
            for _ in range(nchannels)
        ])
        num_steps = len(data)
        self.time_mod = time_mod
        self.w_pos = w_pos
        self.w_nopos = w_nopos
        self.sizes = sizes
        self.p = p
        self.MTGagg = nn.Linear(hids[-1], out_features)
        self.temporal_with_pos = nn.ModuleList([
            TemporalAttention(hids[-1], max_len=num_steps)
            for _ in range(nchannels)
        ])
        self.temporal_no_pos = nn.ModuleList([
            TemporalAttention(hids[-1], max_len=None)
            for _ in range(nchannels)
        ])
        self.cross_attn = nn.ModuleList([
            CrossAttention(hids[-1]) for _ in range(nchannels)
        ])

    def encode(self, nodes):
        all_emb_pos = [[] for _ in range(len(self.SRGA))]
        all_emb_nopos = [[] for _ in range(len(self.SRGA))]
        num_steps = len(data)
        for t in range(num_steps):
            snapshot = data[t]
            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in self.sizes:
                k1 = max(int(size * self.p), 1)
                k2 = size - k1
                if k2 > 0:
                    nbr1 = self.sampler[t](nbr, k1).view(nbr.size(0), k1)
                    nbr2 = self.sampler_t[t](nbr, k2).view(nbr.size(0), k2)
                    nbr = torch.cat([nbr1, nbr2], dim=1).flatten()
                else:
                    nbr = self.sampler[t](nbr, k1).view(-1)
                num_nodes.append(nbr.numel())
                h.append(x[nbr].to(device))
            for idx, layer in enumerate(self.SRGA):
                cond = (idx == 0) or (idx == 1 and t % self.time_mod == 0)
                if cond:
                    out = layer(h, num_nodes)
                    all_emb_pos[idx].append(out)
                    all_emb_nopos[idx].append(out)
        channel_embs = []
        for idx in range(len(self.SRGA)):
            seq = torch.stack(all_emb_pos[idx], dim=0)
            emb_pos = self.temporal_with_pos[idx](seq)
            emb_nopos = self.temporal_no_pos[idx](seq)
            fused = emb_pos * self.w_pos + emb_nopos * self.w_nopos
            channel_embs.append(fused)
        stacked = torch.stack(channel_embs, dim=0).mean(dim=0)
        out = self.MTGagg(stacked)
        neuron.reset_net(self)
        return out

    def forward(self, nodes):
        return self.encode(nodes)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP", help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
                    help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+', default=[512, 64],
                    help='Hidden units for each layer. (default: [512, 64])')
parser.add_argument("--aggr", nargs="?", default="mean", help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage", help="Neighborhood Sampler")
parser.add_argument("--surrogate", nargs="?", default="arctan",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4, help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--alpha', type=float, default=1.0, help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5, help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout probability. (default: 0.6)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=3407, help='Random seed for model. (default: 2024)')
parser.add_argument('--nchannels', type=int, default=2, help='num of LIF models')
parser.add_argument('--cuda', type=str, default='cuda:0', help='which card')
parser.add_argument('--invth', type=float, default=1.0, help='v_threshold')
parser.add_argument('--time_mod', type=int, default=2, help='Mod value for time steps used in conditional computation. (default: 2)')
parser.add_argument('--w_pos', type=float, default=0.6, help='position fuse weights (default: 0.6)')
parser.add_argument('--w_nopos', type=float, default=0.4, help='non-position fuse weights (default: 0.4)')

try:
    args = parser.parse_args()
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - 0.05
    args.val_size = 0.05
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

if args.dataset.lower() == "dblp":
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")

# train:val:test
data.split_nodes(train_size=args.train_size, val_size=args.val_size,
                 test_size=args.test_size, random_state=args.split_seed)

set_seed(args.seed)

device = torch.device(args.cuda)
y = data.y.to(device)

train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                        pin_memory=False, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=False)

model = HASNN(data.num_features, data.num_classes, alpha=args.alpha,
              dropout=args.dropout, sampler=args.sampler, p=args.p,
              aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
              hids=args.hids, act=args.neuron, bias=True, nchannels=args.nchannels, invth=args.invth,
              time_mod=args.time_mod, w_pos=args.w_pos, w_nopos=args.w_nopos).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        loss_fn(model(nodes), y[nodes]).backward()
        optimizer.step()


@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    logits = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, logits, average='macro')
    metric_micro = metrics.f1_score(labels, logits, average='micro')
    return metric_macro, metric_micro

@torch.no_grad()
def extract_embeddings(loader):
    model.eval()
    embeddings = []
    labels = []
    for nodes in loader:
        emb = model.encode(nodes)
        embeddings.append(emb)
        labels.append(y[nodes])
    embeddings = torch.cat(embeddings, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    return embeddings, labels


best_val_metric = test_metric = 0
start = time.time()
for epoch in range(1, args.epochs + 1):
    train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] > best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')
