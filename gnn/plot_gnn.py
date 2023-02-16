from helpers import *
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
import ipywidgets as widgets
from ipywidgets import fixed
import mcbm_dataset
from helpers_custom import *
from helpers_quadrant import *

reload = False
num_samples = None
num_samples_test = None
node_distance = 7
max_num_neighbors = 4

# data = mcbm_dataset.MyDataset(  dataset="test", N = 100, reload=True, \
#                                 radius = node_distance, max_num_neighbors = max_num_neighbors)
data = CreateGraphDataset_quadrant("../data/data_test.root:train", 100, dist = 7)

device = torch.device('cpu')
model = customGNN(graph_iters=5, hidden_size=16).to(device)
model = model.to(torch.float)
model.load_state_dict(torch.load('model_best.pt'))
model.eval()

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

def graph_plot2(input, pred, idx):
    x = input[idx].x[:, 1:3]
    x[:,0] *= 31.0
    x[:,1] *= 71.0
    x = x.numpy()
    y = input[idx].y.numpy()
    e = input[idx].edge_index.numpy()
    plt.figure(figsize=(10, 11.5))
    ax = plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.xlim(0, 31.0)
    plt.ylim(0, 71.0)
    plt.scatter(x[:,0], x[:,1], c=y)
    for i in range(e.shape[1]):
        plt.plot([x[e[0,i],0], x[e[1,i],0]], [x[e[0,i],1], x[e[1,i],1]], 'k-', lw=0.5)

    ax = plt.subplot(1, 2, 2)
    plt.title("Output")
    plt.xlim(0, 31.0)
    plt.ylim(0, 71.0)
    plt.scatter(x[:,0], x[:,1], c=pred[idx], vmin=0.0, vmax=1.0)
    plt.colorbar()
    
    
    plt.show()

data_in = data[0:50]
data_out = []

for i in range(len(data_in)):
    d = data_in[i].to(device)
    out = model(d.x, d.edge_index).detach().cpu().numpy().tolist()
    data_out.append(out)

# print(data_out)
# graph_plot(data, idx)
ip = widgets.interact(  graph_plot2, input=fixed(data_in), \
                        pred=fixed(data_out), idx=(0, len(data_in)-1, 1))

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




