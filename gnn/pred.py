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
import onnxruntime as ort
from helpers import *
from helpers_custom import *
import mcbm_dataset

# data = CreateGraphDataset("../data.root:train", 1000)
data = mcbm_dataset.MyDataset(dataset="train", N = 1000, reload=True)
pred_loader = DataLoader(data, batch_size=1)

device = torch.device('cpu')
model = customGNN().to(device)
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
                example_data.edge_index.detach())
print(input_data[0].grad)
print(input_data[1].grad)
# # for param in model.parameters:
# #     param.cpu().detach()

# dynamic_axes = {'nodes': {0: 'num_nodes'}, 'edge_index': {1: 'index_num_edges'}, "edge_attr": {0: 'atrr_num_edges'}}
# torch.onnx.export(model, input_data, 'gnn.onnx', input_names=["nodes", "edge_index", "edge_attr"], output_names=["output"], export_params=True, dynamic_axes=dynamic_axes, opset_version=16, verbose=True)

# print(model(input_data[0], input_data[1], input_data[2]))
# sess = onnxruntime.InferenceSession('./gnn.onnx', None)
# out_ort = sess.run(None, {'nodes': input_data[0].numpy(), 'edge_index': input_data[1].numpy(), 'edge_attr': input_data[2].numpy()})
# print(out_ort)

ONNX_FILE_PATH = "custom_gnn.onnx"
dynamic_axes = {"nodes": [0, 1], "edge_index": [0, 1]}
torch.onnx.export(model, input_data, ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

expected = model(input_data[0], input_data[1])
options = ort.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1 # 1
options.inter_op_num_threads = 4
# options.enable_profiling=True
options.execution_mode.ORT_SEQUENTIAL
session = ort.InferenceSession(ONNX_FILE_PATH, sess_options=options)
x = [data[i].x.numpy() for i in range(1000)]
edge_index = [data[i].edge_index.numpy() for i in range(1000)]
start = time.time()
for i in range(1000):
    a = session.run(None, {"nodes": x[i], "edge_index": edge_index[i]})[0]
end = time.time()
print("Average time of inference x: ", (end - start) / (1000) * 1000, "ms/sample")

