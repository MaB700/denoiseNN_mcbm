from tracemalloc import start
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch.nn import Sequential, Linear, ReLU, ModuleList, Sigmoid
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
from torch_geometric.nn import SAGEConv
import torch_geometric.nn as geonn
import uproot
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import wandb

def accuracy(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    correct = (y_pred == y_true).sum().item()
    return correct / y_true.size(0)

def CreateGraphDataset(path, n, dist = 7):
    data = uproot.open(path)
    graphs = []
    for batch in tqdm(data.iterate(["time", "tar"], step_size=1000, entry_stop=None if n==0 else n)):
        time_batch = np.array(batch["time"])
        tar_batch = np.array(batch["tar"])
        graphs += [make_graph(i, time_batch, tar_batch, dist) for i in range(len(time_batch))]
    
    return graphs

def make_graph(index, time, tar, dist):
    n = np.count_nonzero(time[index, :] > 0.0001)
    hit_indices = np.nonzero(time[index, :] > 0.0001)
    hits = time[index, :][hit_indices]
    y_pos, x_pos = np.divmod(hit_indices, 32)

    # Nodes
    x = np.zeros((n, 3))
    x[:, 0] = hits.astype('float32') # time [0,1]
    x[:, 1] = (x_pos.astype('float32'))/31.0 # x_coord [0,1]
    x[:, 2] = (y_pos.astype('float32'))/71.0 # y_coord [0,1]
    
    # Edges
    start_index = [] #np.empty((0), dtype=np.int_)
    end_index = [] #np.empty((0), dtype=np.int_)

    for i in range(n):
        for j in range(n):
            if i == j :
                continue
            if abs(x_pos[0, i]-x_pos[0, j]) <= dist and abs(y_pos[0, i]-y_pos[0, j]) <= dist :
                if -0.20 < (x[i, 0] - x[j, 0]) < 0.20 :
                    start_index.append(i)
                    end_index.append(j)
            
            # r = (x[i, 1]-x[j, 1])**2 + ((x[i, 2]-x[j, 2])*2.3 )**2 # 71/31 ~ 2.3
            # t = abs(x[i, 0] - x[j, 0])
            # if r < 0.04 and t < 0.20 : 
            #     start_index = np.append(start_index, i)
            #     end_index = np.append(end_index, j)
    start_index = np.asarray(start_index)
    end_index = np.asarray(end_index)
    edge_index = np.row_stack((start_index, end_index))
    edge_index = torch.from_numpy(edge_index).long()

    # Edge features
    edge_features = np.zeros((edge_index.shape[1], 2))
    if n > 1 :
        for i in range(edge_index.shape[1]):
            x0 = x[start_index[i], 1]
            y0 = x[start_index[i], 2]
            x1 = x[end_index[i], 1]
            y1 = x[end_index[i], 2]
            edge_features[i, 0] = math.sqrt(((x1-x0)*31.0)**2 + ((y1-y0)*71)**2)/(dist*1.41422)
            edge_features[i, 1] = abs(x[start_index[i], 0] - x[end_index[i], 0])
        
    edge_features = torch.from_numpy(edge_features).float()

    # Labels
    y = np.zeros((n, 1))
    y[:, 0] = tar[index, :][hit_indices]


    return Data(x=torch.from_numpy(x).float(), edge_index=edge_index, edge_attr=edge_features, y=torch.from_numpy(y).float())


def MLP(channels, batch_norm=False):
        return Sequential(*[
            Sequential(Linear(channels[i - 1], channels[i]),
                ReLU())
                # nn.BatchNorm1d(channels[i]) if batch_norm else nn.Identity)
            for i in range(1, len(channels))
        ])

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(3, 16, project=True, aggr="add")
        self.conv2 = SAGEConv(16, 16, project=True, aggr="add")
        self.conv3 = SAGEConv(16, 1, project=True, aggr="add")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index).sigmoid()

class Net(torch.nn.Module):
    def __init__(self, data, hidden_nodes):
        super(Net, self).__init__()
        self.k = 8
        self.aggr = "max"
        self.node_encoder = nn.Linear(data.x.size(-1), hidden_nodes)
        #self.edge_encoder = nn.Linear(data.edge_attr.size(-1), hidden_nodes)
        
        self.conv1 = geonn.DynamicEdgeConv(MLP([2*hidden_nodes, hidden_nodes, hidden_nodes]), k=self.k)
        self.conv2 = geonn.DynamicEdgeConv(MLP([2*hidden_nodes, hidden_nodes, hidden_nodes]), k=self.k)
        # self.conv3 = geonn.DynamicEdgeConv(MLP([2*hidden_nodes, hidden_nodes, hidden_nodes]), k=self.k, aggr=self.aggr)
        # self.conv4 = geonn.DynamicEdgeConv(MLP([2*hidden_nodes, hidden_nodes, hidden_nodes]), k=self.k, aggr=self.aggr)
        self.conv5 = geonn.DynamicEdgeConv(MLP([2*hidden_nodes, hidden_nodes, 1]), k=self.k)         

        # self.double()

    def forward(self, x, edge_index, batch):        
        x = self.node_encoder(x)
        #edge_attr = self.edge_encoder(edge_attr)

        x = F.relu(self.conv1(x, batch))
        x = F.relu(self.conv2(x, batch))
        # x = F.relu(self.conv3(x, batch))
        # x = F.relu(self.conv4(x, batch))
        return torch.sigmoid(self.conv5(x, batch))

class LogWandb():
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred
        
        self.log_all()

    def log_all(self):
        cut_value = 0.5        
        auc = roc_auc_score(self.gt, self.pred)
        wandb.log({"test_auc": auc})
        
        tn, fp, fn, tp = confusion_matrix(y_true= self.gt > cut_value, \
                                            y_pred= self.pred > cut_value).ravel()

        wandb.log({"test_acc": (tp+tn)/(tn+fp+tp+fn)}) # accuracy
        wandb.log({"test_sens": tp/(tp+fn)}) # sensitifity
        wandb.log({"test_spec": tn/(tn+fp)}) # specificity
        wandb.log({"test_prec": tp/(tp+fp)}) # precision
        
        wandb.log({"cm": wandb.plot.confusion_matrix(   probs=None,
                                            y_true=[1 if a_ > cut_value else 0 for a_ in self.gt],
                                            preds=[1 if a_ > cut_value else 0 for a_ in self.pred],
                                            class_names=["noise hit", "true hit"],
                                            title="CM")})
        
        len_roc = len(self.gt) if len(self.gt) <= 10000 else 10000
        wandb.log({"roc": wandb.plot.roc_curve( self.gt[:len_roc], 
                                    np.concatenate(((1-self.pred[:len_roc]).reshape(-1,1),self.pred[:len_roc].reshape(-1,1)),axis=1),
                                    classes_to_plot=[1],
                                    title="ROC")})

# class EdgeModel(torch.nn.Module):
#     def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True):
#         super().__init__()

#         self.residuals = residuals

#         layers = [Linear(node_in*2 + edge_in, hid_channels),
#                   ReLU(),
#                   Linear(hid_channels, edge_out)]

#         self.edge_mlp = Sequential(*layers)
#         self.double()

#     def forward(self, src, dest, edge_attr, u, batch):
#         out = torch.cat([src, dest, edge_attr], dim=1)
#         out = self.edge_mlp(out)
#         if self.residuals:
#             out += edge_attr
#         return out

# class NodeModel(torch.nn.Module):
#     def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True):
#         super().__init__()

#         self.residuals = residuals

#         layers = [Linear(node_in + edge_out, hid_channels),
#                   ReLU(),
#                   Linear(hid_channels, node_out)]

#         self.node_mlp = Sequential(*layers)
#         self.double()

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         row, col = edge_index
#         out = edge_attr

#         out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
#         out = torch.cat([x, out1], dim=1)

#         out = self.node_mlp(out)
#         if self.residuals:
#             out += x
#         return torch.sigmoid(out)

# class TestNet(torch.nn.Module):
#     def __init__(self, data, hidden_nodes):
#         super(TestNet, self).__init__()

#         self.node_in = data.x.size(-1) # node features
#         self.edge_in = data.edge_attr.size(-1) # edge features
#         self.hidden_nodes = hidden_nodes
#         node_out = self.hidden_nodes
#         edge_out = self.hidden_nodes
#         lin_nodes = self.hidden_nodes

#         layers = []

#         layer = MetaLayer(node_model=NodeModel(self.node_in, 1, None, edge_out, lin_nodes, residuals=False), 
#         edge_model=EdgeModel(self.node_in, None, self.edge_in, edge_out, lin_nodes, residuals=False))

#         layers.append(layer)
#         self.layers = ModuleList(layers)

#     def forward(self, x, edge_index, edge_attr):#data FIXME:
#         #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         for layer in self.layers:
#             x, edge_attr, _ = layer(x, edge_index, edge_attr, None)#, data.batch  

#         return x
