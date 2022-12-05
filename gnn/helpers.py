from tracemalloc import start
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch_geometric.nn import GCNConv, ARMAConv, GENConv, GeneralConv, global_mean_pool
import torch_geometric.nn as geonn
import uproot




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
            if (x_pos[0, i]-x_pos[0, j]) <= dist and (y_pos[0, i]-y_pos[0, j]) <= dist :
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
    edge_features = np.zeros((edge_index.shape[1], 1))
    if n > 1 :
        for i in range(edge_index.shape[1]):
            x0 = x[start_index[i], 1]
            y0 = x[start_index[i], 2]
            x1 = x[end_index[i], 1]
            y1 = x[end_index[i], 2]
            edge_features[i, 0] = math.sqrt(((x1-x0)*31.0)**2 + ((y1-y0)*71)**2)/(dist*1.41422)
        
    edge_features = torch.from_numpy(edge_features).float()

    # Labels
    y = np.zeros((n, 1))
    y[:, 0] = tar[index, :][hit_indices]


    return Data(x=torch.from_numpy(x).float(), edge_index=edge_index, edge_attr=edge_features, y=torch.from_numpy(y).float())


class Net(torch.nn.Module):
    def __init__(self, data, hidden_nodes):
        super(Net, self).__init__()

        self.node_encoder = nn.Linear(data.x.size(-1), hidden_nodes)
        self.edge_encoder = nn.Linear(data.edge_attr.size(-1), hidden_nodes)
        
        self.conv1 = geonn.GENConv(hidden_nodes, hidden_nodes)
        self.conv2 = geonn.GENConv(hidden_nodes, hidden_nodes)
        self.conv3 = geonn.GENConv(hidden_nodes, hidden_nodes)
        self.conv4 = geonn.GENConv(hidden_nodes, 1)        

        # self.double()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        return torch.sigmoid(self.conv4(x, edge_index, edge_attr))












# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = ARMAConv(3, 32)
#         self.conv2 = ARMAConv(32, 32)
#         self.conv3 = ARMAConv(32, 32)
#         self.conv4 = ARMAConv(32, 1)
#         # self.float()
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))
        
#         # x = F.dropout(x, training=self.training)
#         x = self.conv4(x, edge_index)

#         return torch.sigmoid(x)