from helpers import *
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
import ipywidgets as widgets
from ipywidgets import fixed
import mcbm_dataset

from helpers_quadrant import *

r = 7.0
max_neighbours = 8

def graph_plot(data, idx):
    x = data[idx].x[:, 1:3]
    x[:,0] *= 31.0
    x[:,1] *= 71.0
    x = x.numpy()
    y = data[idx].y.numpy()
    e = data[idx].edge_index.numpy()
    plt.figure(figsize=(5, 11.5))
    plt.xlim(0, 31.0)
    plt.ylim(0, 71.0)
    plt.scatter(x[:,0], x[:,1], c=y)
    for i in range(e.shape[1]):
        plt.plot([x[e[0,i],0], x[e[1,i],0]], [x[e[0,i],1], x[e[1,i],1]], 'k-', lw=0.5)
        # plt.plot(x[e[:,i],0], x[e[:,i],1], 'k-', lw=0.5)
    # plt.show(block=True)
    plt.show()

# data = mcbm_dataset.MyDataset(dataset="test", N = 40, reload=True, radius=r, max_num_neighbors=max_neighbours)
data = CreateGraphDataset_quadrant("../data/data.root:train", 40, dist = 7)
print(data[0].edge_index.numpy())
#idx = 11
tot_edge = 0
for i in range(len(data)):
    tot_edge += data[i].edge_index.shape[1]
print(data[0])
print("tot_edge: ", tot_edge)
# graph_plot(data, idx)
ip = widgets.interact(graph_plot, data=fixed(data), idx=(0, len(data)-1, 1))

# x = data[idx].x[:, 1:3]
# x[:,0] *= 31.0
# x[:,1] *= 71.0
# x = x.numpy()
# y = data[idx].y.numpy()
# e = data[idx].edge_index.numpy()
# # plot x in 2D with x[:,0] as x-axis and x[:,1] as y-axis and color the nodes according to their label y
# # also plot lines between nodes that are connected by an edge, start and end node index are given in e
# plt.figure(figsize=(5, 11.5))
# plt.xlim(0, 31.0)
# plt.ylim(0, 71.0)
# plt.scatter(x[:,0], x[:,1], c=y)
# for i in range(e.shape[1]):
#     plt.plot([x[e[0,i],0], x[e[1,i],0]], [x[e[0,i],1], x[e[1,i],1]], 'k-', lw=0.5)
#     # plt.plot(x[e[:,i],0], x[e[:,i],1], 'k-', lw=0.5)
# plt.show(block=True)




