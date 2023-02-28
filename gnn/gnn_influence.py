import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import fixed
import torch
from torch_geometric.loader import DataLoader

from helpers_custom import *
from helpers_quadrant import *
import mcbm_dataset

reload = False
num_samples = 50
node_distance = 12
max_num_neighbors = 32

batch_size = 256
lr = 1e-3
max_epochs = 100
es_patience = 5

# data = mcbm_dataset.MyDataset(  dataset="test", N = num_samples, reload=reload, \
#                                 radius = node_distance, max_num_neighbors = max_num_neighbors)
data = CreateGraphDataset_quadrant("../data/data_test.root:train", 40, dist = 7)
val_loader = DataLoader(data, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = customGNN(graph_iters=5, hidden_size=16).to(device)
# model = Net(train_dataset[0], 32).to(device)
model = model.to(torch.float)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.load_state_dict(torch.load('model_best.pt', map_location=device))

def jacobi(data):
    data = data.to(device)
    data.requires_grad_(*['x'], True)
    dx = torch.autograd.functional.jacobian(lambda t: model(t, data.edge_index), data.x, create_graph=True)
    return dx

def influence(data, x = 0, abs = False): # return I_x(y) for all y
    if x < 0: x = 0
    if x >= data.x.size(dim=0): x = data.x.size(dim=0)    
    dx = jacobi(data) # matrix (output_feature_size x input_feature_size)
    if abs: 
        nom = torch.sum(torch.abs(dx[x]), dim=2)
        denom = torch.sum(torch.abs(dx[x]), dim=None)
    else:
        nom = torch.sum(dx[x], dim=2)
        denom = torch.sum(dx[x], dim=None)
    return torch.divide(nom, denom)[0] #TODO: check if denom is colse to zero

def graph_plot(data, idx, x_in):
    x = data[idx].x[:, 1:3]
    x[:,0] *= 31.0
    x[:,1] *= 71.0
    x = x.numpy()
    y = data[idx].y.numpy()
    e = data[idx].edge_index.numpy()
    inf = influence(data[idx], x_in, abs=False).detach().cpu().numpy()
    plt.figure(figsize=(5, 11.5))
    plt.xlim(0, 31.0)
    plt.ylim(0, 71.0)
    max_inf = max(inf[np.arange(len(inf)) != x_in])
    plt.scatter(x[:,0], x[:,1], c=inf, vmin=0, vmax=max_inf)
    plt.scatter(x[x_in,0], x[x_in,1], c='r', marker="s", s=55)
    plt.clim(0, max_inf)
    plt.colorbar()
    for i in range(e.shape[1]):
        plt.plot([x[e[0,i],0], x[e[1,i],0]], [x[e[0,i],1], x[e[1,i],1]], 'k-', lw=0.2)
        # plt.plot(x[e[:,i],0], x[e[:,i],1], 'k-', lw=0.5)
    plt.show(block=True)
    # plt.show()

# ip = widgets.interact(graph_plot, data=fixed(data), idx=(0, len(data)-1, 1), x_in=(0, 5, 1))
# print(influence(data[0], 5, abs=True).detach().cpu().numpy())
graph_plot(data, 22, 8)