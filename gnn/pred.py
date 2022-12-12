# %%
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
import onnxruntime
from helpers import *

data = CreateGraphDataset("../data.root:train", 1000)
pred_loader = DataLoader(data, batch_size=1)

device = torch.device('cpu')
model = TestNet(data[0], 8).to(device)
model = model.to(torch.float)
model.load_state_dict(torch.load('model_best.pt', map_location=device))
model.eval()

# for param in model.parameters():
#     print(type(param), param.size())
#     print(param.grad)

def predict(loader):
    model.eval()
    #tar = np.empty((0))
    #prd = np.empty((0))
    for data in loader :
        data = data.to(device)
        #start = time.time()
        pred = model(data).cpu().detach().numpy()
        print(pred)
        break
        #end = time.time()
        #print(end - start)
        #target = data.y.cpu().detach().numpy()
        #tar = np.append(tar, target)
        #prd = np.append(prd, np.array(pred))
    #return tar, prd

# pr = cProfile.Profile()
# pr.enable()
# #gt, pred = 
# predict(pred_loader)
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# #ps.print_stats()
# filename = 'profile.prof'
# pr.dump_stats(filename)
# print(s.getvalue())
example_data = data[0]
input_data = (  example_data.x.detach(), 
                example_data.edge_index.detach(), 
                example_data.edge_attr.detach())
print(input_data[0].grad)
print(input_data[1].grad)
print(input_data[2].grad)
# # for param in model.parameters:
# #     param.cpu().detach()

dynamic_axes = {'nodes': {0: 'num_nodes'}, 'edge_index': {1: 'index_num_edges'}, "edge_attr": {0: 'atrr_num_edges'}}
torch.onnx.export(model, input_data, 'gnn.onnx', input_names=["nodes", "edge_index", "edge_attr"], output_names=["output"], export_params=True, dynamic_axes=dynamic_axes, opset_version=16, verbose=True)

print(model(input_data[0], input_data[1], input_data[2]))
sess = onnxruntime.InferenceSession('./gnn.onnx', None)
out_ort = sess.run(None, {'nodes': input_data[0].numpy(), 'edge_index': input_data[1].numpy(), 'edge_attr': input_data[2].numpy()})
print(out_ort)
# %%
