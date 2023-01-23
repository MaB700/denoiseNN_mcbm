import os
import numpy as np
import uproot
import time
import mcbm_dataset
# import helpers

samples = 1000
start = time.time()
data = mcbm_dataset.MyDataset(dataset="train", N = samples, reload=True, undirected=True)
#data = helpers.CreateGraphDataset("../data/data.root:train", 100000, dist = 5)
print(data[0])
print("Time to load data: ", time.time()-start)

# print(os.path.abspath(__file__ + "/../data"))
# print(os.path.dirname(__file__))
# print(os.path.realpath('../data'))
# print(os.path.realpath('E:\git\denoiseNN_mcbm\data'))

