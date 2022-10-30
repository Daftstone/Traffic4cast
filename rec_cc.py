import os
import argparse

import statistics
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch_geometric
import tqdm
from torch import nn
from pathlib import Path
import numpy as np
import random

import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset

t4c_apply_basic_logging_config(loglevel="DEBUG")

BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default="london", help="london, melbourne, madrid")  # city
parser.add_argument('--split', type=float, default=0.8, help="train:test=0.8:0.2")

parser.add_argument('--model_state', type=str, default="train")

parser.add_argument('--fill', type=int, default=1)
parser.add_argument('--normalize', type=str, default="zs")

parser.add_argument('--hidden_channels', type=int, default=32, help="hidden_channels")
parser.add_argument('--num_layers', type=int, default=3, help="num_layers for predict model")

parser.add_argument('--batch_size', type=int, default=2, help="batch_size")
parser.add_argument('--epochs', type=int, default=20, help="epochs")

parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
parser.add_argument('--num_edge_classes', type=int, default=3, help="num_edge_classes")
parser.add_argument('--num_features', type=int, default=4, help="num_features")
parser.add_argument('--cluster', type=int, default=0, help="num_features")

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--device', type=int, default=0, help="available cuda device")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    print("==================== Using CUDA %d ===================" % args.device)
else:
    device = torch.device("cpu")
    print("===================== Using CPU =====================")

opt = vars(args)
opt[
    "save_path"] = f"./save/cc/class_L{str(opt['num_layers'])}_H{str(opt['hidden_channels'])}_F{str(opt['fill'])}_N{opt['normalize']}_B{str(opt['batch_size'])}_e{str(opt['epochs'])}/{opt['city']}/"
opt[
    "submission_name"] = f"cc/class_{str(opt['num_layers'])}{str(opt['hidden_channels'])}{opt['fill']}{opt['normalize']}{str(opt['batch_size'])}{str(opt['epochs'])}"


class RecLinear(nn.Module):
    def __init__(self, num_edges, num_nodes, num_attrs, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):

        super(RecLinear, self).__init__()

        self.embed = nn.Embedding(num_edges, hidden_channels)
        self.node_embed = nn.Embedding(num_nodes, hidden_channels)
        self.node_embed1 = nn.Embedding(num_nodes, 4)
        self.time_embed = nn.Embedding(96, hidden_channels)
        self.week_embed = nn.Embedding(7, hidden_channels)
        self.node_index = torch.arange(0, num_nodes).to(device)

        self.node_lin = nn.Linear(in_channels, hidden_channels)
        self.node_lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.attr_lin = nn.Linear(num_attrs, hidden_channels)
        self.attr_lin1 = nn.Sequential(nn.Linear(num_attrs, hidden_channels), nn.LeakyReLU(),
                                       nn.Linear(hidden_channels, hidden_channels))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(hidden_channels * 6, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.fc1 = nn.Linear(num_nodes, 256)
        self.fc2 = nn.Linear(256, 32)  # 均值 向量
        self.fc3 = nn.Linear(256, 32)  # 保准方差 向量
        self.fc4 = nn.Linear(32, 256)
        self.fc5 = nn.Linear(256, num_nodes)

        from torch_geometric.nn import GATv2Conv as GCNConv

        self.conv1 = torch.nn.ModuleList()
        for i in range(3):
            self.conv1.append(GCNConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))

        self.conv2 = torch.nn.ModuleList()
        for i in range(3):
            self.conv2.append(GCNConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))

        self.gcn_lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.gcn_lin2 = nn.Linear(hidden_channels * 2, hidden_channels)

    def gelu(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    # 编码过程
    def encode(self, x):
        h = self.gelu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = self.gelu(self.fc4(z))
        h = self.fc5(h)
        return h

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.node_embed.reset_parameters()
        self.node_embed1.reset_parameters()
        self.time_embed.reset_parameters()
        self.node_lin.reset_parameters()
        self.node_lin1.reset_parameters()
        self.attr_lin.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc5.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.conv1:
            lin.reset_parameters()
        for lin in self.conv2:
            lin.reset_parameters()
        self.gcn_lin1.reset_parameters()
        self.gcn_lin2.reset_parameters()

    def forward(self, index, edge_index, x, attr, cur_t, cur_w):
        mask_idx = (torch.sum(x, dim=1, keepdim=True) != nan_to_num_map[opt['city']] * 4).type(torch.float)

        xmax = 23.91
        xmin = nan_to_num_map[opt['city']]
        x_norm = (x - xmin) / (xmax - xmin)

        ratio = 0.8 + 0.4 * np.random.rand(1)[0]

        x_norm = x_norm * ratio

        drop_idx = (torch.rand_like(x_norm[:, 0:1]) > 0.4).type(torch.float)
        x_norm = x_norm * drop_idx

        x_norm = torch.transpose(x_norm, 0, 1)
        mu, log_var = self.encode(x_norm)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)

        x_rec = x_rec / ratio

        x_rec = torch.transpose(x_rec, 0, 1)
        x_rec = x_rec * (xmax - xmin) + xmin
        x_rec1 = mask_idx * x + (1 - mask_idx) * x_rec

        attr1 = self.attr_lin(attr)
        embed = self.embed(index)

        node_embed = self.node_embed(self.node_index)
        pre_data = node_embed
        for conv in self.conv1:
            node_embed = conv(node_embed, edge_index, attr1)
            node_embed = self.gelu(node_embed) + pre_data

        data = self.gelu(self.node_lin(x_rec1))
        pre_data = data
        for conv in self.conv2:
            data = conv(data, edge_index, attr1)
            data = self.gelu(data) + pre_data

        x_i = torch.index_select(data, 0, edge_index[0])
        x_j = torch.index_select(data, 0, edge_index[1])
        x = torch.concat([x_i, x_j], dim=1)
        x = self.gcn_lin1(x)

        x_i = torch.index_select(node_embed, 0, edge_index[0])
        x_j = torch.index_select(node_embed, 0, edge_index[1])
        x1 = torch.concat([x_i, x_j], dim=1)
        x1 = self.gcn_lin2(x1)

        time_embed = self.time_embed(cur_t.long())
        week_embed = self.week_embed(cur_w.long())

        xf = torch.cat([embed, self.attr_lin1(attr), x, x1, time_embed, week_embed], dim=1)

        for lin in self.lins[:-1]:
            xf = lin(xf)
            xf = self.gelu(xf)

        xf = self.lins[-1](xf)

        return xf, x_rec


# data loader
dataset = T4c22GeometricDataset(root=BASEDIR, city=opt['city'],
                                edge_attributes=["speed_kph", "parsed_maxspeed", "length_meters", "counter_distance",
                                                 "importance", "highway", "oneway", ], split="train", fill=opt['fill'],
                                normalize=opt['normalize'], cachedir=Path(f"{BASEDIR}/cache"), idx=opt['cluster'])
test_dataset = T4c22GeometricDataset(root=BASEDIR, city=opt['city'],
                                     edge_attributes=["speed_kph", "parsed_maxspeed", "length_meters",
                                                      "counter_distance", "importance", "highway", "oneway", ],
                                     split="test", fill=opt['fill'],
                                     normalize=opt['normalize'], cachedir=Path(f"{BASEDIR}/cache"), idx=opt['cluster'])
print("################## Data Information #################")
print("Dataset Size\t", len(dataset))
print("Test Dataset Size\t", len(test_dataset))
print("The statistics of training set are: Min [%d]\tMax [%d]\tMean [%.4f]\tStd[%.4f]" % (
    dataset.min_volume, dataset.max_volume, dataset.mean_volume, dataset.std))
# print(dataset.get(0))

# split dataset
spl = int(((opt['split'] * len(dataset)) // 2) * 2)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [spl, len(dataset) - spl])
print("Train Dataset Size\t", len(train_dataset))
print("Validation Dataset Size\t", len(val_dataset))

# city class fraction
city_class_fractions = class_fractions[opt['city']]
city_class_weights = torch.tensor(
    get_weights_from_class_fractions(
        [city_class_fractions['green'], city_class_fractions['yellow'],
         city_class_fractions['red']])).float()
print("City Class Weight\t", city_class_weights)
print("######################## End ########################")

nan_to_num_map = {"london": -1.21, "melbourne": -0.8, "madrid": -0.56}

if __name__ == "__main__":

    city_class_weights = city_class_weights.to(device)
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    num_edges = edge_index.shape[1]
    num_attrs = edge_attr.shape[1]
    num_nodes = np.max(edge_index.cpu().numpy()) + 1
    print('num_nodes', num_nodes)

    np.save(f"temp/{opt['city']}_edge_attr.npy", edge_attr.cpu().numpy())

    index = torch.arange(0, num_edges).to(device)

    if not os.path.exists(opt['save_path']):
        os.makedirs(opt['save_path'])

    model = RecLinear(num_edges, num_nodes, num_attrs, opt['num_features'], opt['hidden_channels'],
                      opt['num_edge_classes'],
                      opt['num_layers'], opt['dropout']).to(device)

    if opt['model_state'] == "train":

        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=5, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            model.reset_parameters()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
            loss_mse = torch.nn.MSELoss()

            min_loss = 10000

            print("fold", fold)
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            for epoch in tqdm.tqdm(range(1, 21), "epochs", total=opt['epochs']):

                model.train()

                losses = []
                optimizer.zero_grad()

                pbar = tqdm.tqdm(
                    torch_geometric.loader.dataloader.DataLoader(dataset, batch_size=opt['batch_size'],
                                                                 num_workers=8, sampler=train_subsampler),
                    "train",
                    total=len(train_dataset) // opt['batch_size'], )
                count = 0
                for data in pbar:
                    data = data.to(device)
                    data.x[data.x > 23.91] = 23.91
                    data.x[data.x == -1] = nan_to_num_map[opt['city']]
                    loss = 0.
                    if (count == 0):
                        lens = data.x.shape[0] // opt['batch_size']
                        lens1 = data.y.shape[0] // opt['batch_size']
                        count += 1

                    for i in range(data.y.shape[0] // lens1):
                        t = data.t[i]
                        cur_t = torch.ones_like(edge_index[0]) * t
                        week = data.week[i]
                        cur_week = torch.ones_like(edge_index[0]) * week

                        y = data.y[i * lens1:(i + 1) * lens1].nan_to_num(-1)
                        x = data.x[i * lens:(i + 1) * lens]
                        y_hat, x_rec = model(index, edge_index, x,
                                             edge_attr, cur_t, cur_week)
                        y = y.long()

                        train_index = torch.nonzero(torch.sum(x, dim=1) != nan_to_num_map[opt['city']] * 4).squeeze()

                        rec_loss = loss_mse(x[train_index], x_rec[train_index])
                        acc_loss = loss_f(y_hat, y)

                        loss += rec_loss + acc_loss

                    loss = loss / (data.y.shape[0] // lens1)
                    loss.backward()

                    # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(acc_loss.cpu().item())

                    pbar.set_postfix(rec_loss=rec_loss.cpu().item(), acc_loss=acc_loss.cpu().item())

                print(f"train_loss={np.mean(losses)} after epoch {epoch}")

                model.eval()

                losses = []
                for data in tqdm.tqdm(
                        torch_geometric.loader.dataloader.DataLoader(dataset, batch_size=opt['batch_size'],
                                                                     num_workers=8,
                                                                     sampler=test_subsampler), "test",
                        total=len(val_dataset) // opt['batch_size']):
                    data = data.to(device)
                    data.x[data.x > 23.91] = 23.91
                    data.x[data.x == -1] = nan_to_num_map[opt['city']]
                    loss = 0
                    for i in range(data.y.shape[0] // lens1):
                        t = data.t[i]
                        cur_t = torch.ones_like(edge_index[0]) * t
                        week = data.week[i]
                        cur_week = torch.ones_like(edge_index[0]) * week

                        x = data.x[i * lens:(i + 1) * lens]
                        y_pred = 0.
                        for iii in range(5):
                            y_hat, _ = model(index, edge_index, x,
                                             edge_attr, cur_t, cur_week)
                            y_pred += y_hat.detach().cpu().numpy()
                        y = data.y[i * lens1:(i + 1) * lens1].nan_to_num(-1)
                        y = y.long()

                        y_pred /= 5.

                        loss += loss_f(torch.tensor(y_pred).to(device), y)
                    loss = loss / (data.y.shape[0] // lens1)
                    losses.append(loss.cpu().item())

                print(f"val_loss={np.mean(losses)} after epoch {epoch}")

                if (np.mean(losses) < min_loss):
                    min_loss = np.mean(losses)
                    torch.save(model.state_dict(), f"{opt['save_path']}model_best_%d.pt" % fold)
                torch.save(model.state_dict(), f"{opt['save_path']}model_{fold}_{epoch:03d}.pt")
    else:

        model.eval()

        dfs = []
        for idx, data in tqdm.tqdm(enumerate(test_dataset), total=len(test_dataset)):
            data = data.to(device)
            data.x[data.x > 23.91] = 23.91
            data.x[data.x == -1] = nan_to_num_map[opt['city']]

            t = data.t
            cur_t = torch.ones_like(edge_index[0]) * t
            week = data.week
            cur_week = torch.ones_like(edge_index[0]) * week

            if (opt['city'] == 'melbourne'):
                y_pred = 0.
                for fold in range(5):
                    model.load_state_dict(torch.load(f"{opt['save_path']}model_best_{fold}.pt"))
                    model.eval()
                    for ii in range(5):
                        y_hat, _ = model(index, edge_index, data.x, edge_attr, cur_t, cur_week)
                        y_hat = y_hat.detach()
                        y_pred += y_hat
                y_pred /= 25.
            else:
                y_pred = 0.
                for fold in range(5):
                    for mm in range(16, 21):
                        model.load_state_dict(torch.load(f"{opt['save_path']}model_{fold}_{mm:03d}.pt"))
                        model.eval()
                        for ii in range(5):
                            y_hat, _ = model(index, edge_index, data.x, edge_attr, cur_t, cur_week)
                            y_hat = y_hat.detach()
                            y_pred += y_hat
                y_pred /= 125.

            df = test_dataset.torch_road_graph_mapping._torch_to_df_cc(data=y_pred, day="test", t=idx)
            dfs.append(df)
        df = pd.concat(dfs)
        df["test_idx"] = df["t"]
        del df["day"]
        del df["t"]

        submission = df
        print(submission.head(20))

        (BASEDIR / "submissions" / opt['submission_name'] / opt['city'] / "labels").mkdir(exist_ok=True, parents=True)
        table = pa.Table.from_pandas(submission)
        pq.write_table(table, BASEDIR / "submissions" / opt['submission_name'] / opt[
            'city'] / "labels" / f"cc_labels_test.parquet", compression="snappy")
