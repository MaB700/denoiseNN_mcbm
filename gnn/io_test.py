import numpy as np
import time
from joblib import Parallel, delayed
import torch
import torch_geometric.transforms as T

from torch_geometric.data.data import Data
import my_dataset

start = time.time()
data = my_dataset.MyDataset(root="E:/ML_data/test", dataset="test", reload=True)
x = data[0]
print(data[0].is_undirected())
print(data[0])
print(data[0].edge_index)
print("Time to load dataset: ", (time.time()-start), "s")

# num = 100000
# x = np.random.rand(num, 30, 3)
# y = np.random.rand(num, 1)
# pos = np.random.rand(num, 30, 2)
# edge_index = np.random.randint(0, 30-1, (num, 2, 130))

# # same x, y, pos, edge_index to disk
# np.save("E:/ML_data/test/x.npy", x)
# np.save("E:/ML_data/test/y.npy", y)
# np.save("E:/ML_data/test/pos.npy", pos)
# np.save("E:/ML_data/test/edge_index.npy", edge_index)

# del x, y, pos, edge_index

# start = time.time()
# x = np.load("E:/ML_data/test/x.npy")
# y = np.load("E:/ML_data/test/y.npy")
# pos = np.load("E:/ML_data/test/pos.npy")
# edge_index = np.load("E:/ML_data/test/edge_index.npy")
# print("Time to load npy: ", (time.time()-start), "s")

# x = torch.from_numpy(x).float()
# y = torch.from_numpy(y).float()
# pos = torch.from_numpy(pos).float()
# edge_index = torch.from_numpy(edge_index).int()

# start = time.time()
# data = [Data(   x=x[i], 
#                 y=y[i],
#                 # edge_index=torch.from_numpy(edge_index[i]).int(), 
#                 pos=pos[i]) 
#                 for i in range(num)]
# print("Time to create sequentiel: ", (time.time()-start), "s")

# start = time.time()
# data = [T.RadiusGraph(0.5, max_num_neighbors=4)(data[i]) for i in range(num)]
# print("Time to transform: ", (time.time()-start), "s")

# start = time.time()
# torch.save(data, "E:/ML_data/test/data.pt")
# print("Time to save: ", (time.time()-start), "s")

# start = time.time()
# data = torch.load("E:/ML_data/test/data.pt")
# print("Time to load: ", (time.time()-start), "s")