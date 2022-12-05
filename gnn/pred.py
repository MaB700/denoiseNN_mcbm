import torch_geometric
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
import time
import cProfile, pstats, io
from pstats import SortKey
from sklearn.metrics import roc_auc_score, confusion_matrix

from helpers import *

data = CreateGraphDataset("../TMVA_mcbm/data.root:train", 1000)
pred_loader = DataLoader(data, batch_size=1)

device = torch.device('cpu')
model = Net(data[0], 32).to(device)
model = model.to(torch.float)
model.load_state_dict(torch.load('model_best.pt'))
model.eval()


def predict(loader):
    model.eval()
    #tar = np.empty((0))
    #prd = np.empty((0))
    for data in loader :
        data = data.to(device)
        #start = time.time()
        pred = model(data).cpu().detach().numpy()
        #end = time.time()
        #print(end - start)
        #target = data.y.cpu().detach().numpy()
        #tar = np.append(tar, target)
        #prd = np.append(prd, np.array(pred))
    #return tar, prd

pr = cProfile.Profile()
pr.enable()
#gt, pred = 
predict(pred_loader)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
filename = 'profile.prof'
pr.dump_stats(filename)
print(s.getvalue())