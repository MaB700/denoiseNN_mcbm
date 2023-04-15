import time

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import remove_isolated_nodes
import torch.nn as nn

import onnxruntime as ort

from helpers import *
# from memory_profiler import profile

from helpers_custom import *
from helpers_quadrant import *
import mcbm_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"

# print pytorch version
print(torch.__version__)
# model to .onnx works properly with pytorch 1.13
model = customGNN(graph_iters=3, hidden_size=16, num_layers=3)
# print(model)

# data = CreateGraphDataset('../data/data.root:train', 16, 3)
data = CreateGraphDataset_quadrant('../data/data.root:train', 10000, 7)
# data = mcbm_dataset.MyDataset(  dataset="train", N = 16, reload=True, \
#                                 radius = 7, max_num_neighbors = 8)
data_loader_rich = DataLoader(data, batch_size=1)

x = torch.randn(3, 3)
x2 = torch.randn(10, 3)
x3 = torch.randn(35, 3)
edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
edge_index2 = torch.tensor([[1, 0, 2, 2, 2, 4, 3, 7, 6], 
                            [0, 1, 3, 4, 7, 0, 4, 8, 9]])
edge_index3 = torch.tensor([[0, 1, 2, 3, 4, 3, 7, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                            [1, 0, 2, 1, 0, 4, 8, 9, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28]])

n_disconnected_graphs = 16
# data = [Data(x=x3, edge_index=edge_index3) for _ in range(n_disconnected_graphs)]
data_loader = DataLoader(data, batch_size=n_disconnected_graphs)
datax = None
for data_it in data_loader_rich:
    datax = data_it

input_data = (x, edge_index)
input_data2 = (x2, edge_index2)
input_data3 = (data[0].x, data[0].edge_index)
input_datax = (datax.x, datax.edge_index)
print(datax.edge_index.shape)
ONNX_FILE_PATH = "custom.onnx"
dynamic_axes = {"nodes": [0, 1], "edge_index": [0, 1]}

# dynamic_axes = {"nodes": {0: "num_nodes", 1:"node_features"}, "edge_index": {1: "num_edges"}, "output": {0: "num_nodes"}}
torch.onnx.export(model, input_data, ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

expected = model(*input_data)
expected2 = model(*input_data2)
expected3 = model(*input_data3)
expectedx = model(*input_datax)
# print(expected3)
options = ort.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1 # athena uses 1
options.inter_op_num_threads = 1
options.enable_profiling = False
options.execution_mode.ORT_SEQUENTIAL
session = ort.InferenceSession(ONNX_FILE_PATH, sess_options=options)
out = session.run(None, {"nodes": input_data[0].numpy(), "edge_index": input_data[1].numpy()})[0]
out2 = session.run(None, {"nodes": input_data2[0].numpy(), "edge_index": input_data2[1].numpy()})[0]
out3 = session.run(None, {"nodes": input_data3[0].numpy(), "edge_index": input_data3[1].numpy()})[0]
outx = session.run(None, {"nodes": input_datax[0].numpy(), "edge_index": input_datax[1].numpy()})[0]

import time
# start = time.time()
# j = 0
# for i in range(1000):
#     if j == (n_disconnected_graphs - 1): 
#         j = 0
#     a = session.run(None, {"nodes": data[j].x.numpy(), "edge_index": data[j].edge_index.numpy()})[0]
#     j += 1
# end = time.time()
# print("Average time of inference: ", (end - start) / 1000 * 1000, "ms")

x_list = []
edge_index_list = []
for i in range(len(data)):
    x_list.append(data[i].x.detach().numpy())
    edge_index_list.append(data[i].edge_index.detach().numpy())

start = time.time()
for i in range(len(data)):
    a = session.run(None, {"nodes": x_list[i], "edge_index": edge_index_list[i]})[0]
end = time.time()
print("Average time of inference x: ", (end - start) *1000 / len(data), "ms")



from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnxruntime.quantization as oq

class CalibrationDataProvider(CalibrationDataReader):
    def __init__(self):
        super(CalibrationDataProvider, self).__init__()
        self.counter = 0
        self.x = torch.randn(1000, 25, 3).numpy()
        self.edge_index = torch.randint(0, 25, (1000, 2, 40)).numpy()    

    def get_next(self):
        if self.counter > 1000 - 1:
            return None
        else:
            out = {'nodes': self.x[self.counter], 'edge_index': self.edge_index[self.counter]}
            self.counter += 1
            return out

# cdp = CalibrationDataProvider()
# quantized_onnx_model = oq.quantize_static(ONNX_FILE_PATH, './customgnn_quant.onnx', weight_type=oq.QuantType.QInt8, calibration_data_reader=cdp, per_channel=True, reduce_range=True)

# session2 = ort.InferenceSession('./customgnn_quant.onnx', options)
j = 0
# start = time.time()
# for i in range(1000):    
#     if j == (n_disconnected_graphs - 1):
#         j = 0
#     x = session2.run(None, {"nodes": data[j].x.numpy(), "edge_index": data[j].edge_index.numpy()})[0]
#     j += 1
# end = time.time()
# print("Average time of inference ort_quant: ", (end - start) / 1000 * 1000, "ms")

# start = time.time()
# for i in range(1000):
#     x = session2.run(None, {"nodes": input_datax[0].numpy(), "edge_index": input_datax[1].numpy()})[0]
# end = time.time()
# print("Average time of inference ort_quant x: ", (end - start) / 1000 * 1000, "ms", (end - start) / (1000*n_disconnected_graphs) * 1000, "ms/sample")









# x1 = np.expand_dims(x.numpy().flatten(), axis=0)
# x2 = np.expand_dims(x2.numpy().flatten(), axis=0)
# x3 = np.expand_dims(x3.numpy().flatten(), axis=0)
# edge_index1 = np.expand_dims(edge_index.numpy().flatten(order='F'), axis=0)
# edge_index2 = np.expand_dims(edge_index2.numpy().flatten(order='F'), axis=0)
# edge_index3 = np.expand_dims(edge_index3.numpy().flatten(order='F'), axis=0)
# out_pyg1 = np.expand_dims(expected.detach().cpu().numpy().flatten(), axis=0)
# out_pyg2 = np.expand_dims(expected2.detach().cpu().numpy().flatten(), axis=0)
# out_pyg3 = np.expand_dims(expected3.detach().cpu().numpy().flatten(), axis=0)
# out_onnx1 = np.expand_dims(out.flatten(), axis=0)
# out_onnx2 = np.expand_dims(out2.flatten(), axis=0)
# out_onnx3 = np.expand_dims(out3.flatten(), axis=0)

# x = (x1, x2, x3)
# edge_index = (edge_index1, edge_index2, edge_index3)
# out_pyg = (out_pyg1, out_pyg2, out_pyg3)
# out_onnx = (out_onnx1, out_onnx2, out_onnx3)

# write x and edge_index to file with comma as delimiter, each row is a sample
# with open("in_node.csv", "w") as f:
#     for i in range(len(x)):
#         np.savetxt(f, x[i], delimiter=",")

# with open("in_edge_index.csv", "w") as f:
#     for i in range(len(edge_index)):
#         np.savetxt(f, edge_index[i], delimiter=",", fmt="%u")

# with open("out_pyg.csv", "w") as f:
#     for i in range(len(out_pyg)):
#         np.savetxt(f, out_pyg[i], delimiter=",")

# with open("out_onnx.csv", "w") as f:
#     for i in range(len(out_onnx)):
#         np.savetxt(f, out_onnx[i], delimiter=",")

