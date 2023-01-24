from helpers import *
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
import ipywidgets as widgets
from ipywidgets import fixed
import mcbm_dataset

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

data = mcbm_dataset.MyDataset(dataset="test", N = 100, reload=True)
#data = CreateGraphDataset("../data.root:train", 100, dist=5)
#idx = 11
print(data[0])
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




