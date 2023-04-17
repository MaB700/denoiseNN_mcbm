import numpy as np
import torch
from torch_geometric.loader import DataLoader
import uproot
from mcbm_dataset import MyDataset

num_samples = 10000
reload = True
node_distance = 12
max_num_neighbors = 16

data_gnn = MyDataset(dataset="train", N = num_samples, reload=reload, \
                    radius = node_distance, max_num_neighbors = max_num_neighbors)

loader_gnn = DataLoader(data_gnn, batch_size=64)
num_hits_gnn = [0, 0, 0]
for data in loader_gnn:
    mask = (data.y > -0.1)
    true_hits = (data.y >= 0.5)
    noise_hits = (data.y < 0.5)
    num_hits_gnn[0] += torch.sum(mask).numpy()
    num_hits_gnn[1] += torch.sum(true_hits).numpy()
    num_hits_gnn[2] += torch.sum(noise_hits).numpy()    

for h in num_hits_gnn:
   print(h)

print("is_equal_gnn:", num_hits_gnn[0] == (num_hits_gnn[1] + num_hits_gnn[2]))


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, path, samples):
    self.x = None
    self.y = None
    with uproot.open(path) as file:
        self.x = np.reshape(np.array(file["train"]["time"].array(entry_stop=samples)), (-1, 1, 72, 32))    
        self.y = np.reshape(np.array(file["train"]["tar"].array(entry_stop=samples)), (-1, 1, 72, 32))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

data_cnn = MyDataset("../data/data.root", num_samples)

cnn_loader = torch.utils.data.DataLoader(data_cnn, batch_size=64)

num_hits = [0, 0, 0]
for input, target in cnn_loader:
    mask = (input > 0.01)
    true_hits = (target[mask] >= 0.5)
    noise_hits = (target[mask] < 0.5)
    # count number of non-zero elements in input[mask]
    num_hits[0] += torch.sum(mask).numpy()
    num_hits[1] += torch.sum(true_hits).numpy()
    num_hits[2] += torch.sum(noise_hits).numpy()

for h in num_hits:
   print(h)

print("is_equal_cnn:", num_hits[0] == (num_hits[1] + num_hits[2]))
