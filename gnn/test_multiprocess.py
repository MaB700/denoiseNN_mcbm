import numpy as np
import time
from multiprocessing import Process, Queue
from queue import PriorityQueue
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import onnxruntime as ort

from helpers import *
from helpers_custom import *
from test_multiprocess_helpers import *
print(torch.__version__)

num_cores, processor_list = cpu_count_physical()
print("num_cores: ", num_cores, "processor_list: ", processor_list)

threads_per_instance = 2
# num_instances = num_cores // threads_per_instance
num_instances = 1
print("num_instances: ", num_instances)

device = torch.device("cpu")
model = customGNN().to(device)
model = model.to(torch.float)

data0 = CreateGraphDataset('../data.root:train', 1000, 3)
data_loader = DataLoader(data0, batch_size=1)

input_data = (data0[0].x, data0[0].edge_index)
print(data0[0])
ONNX_FILE_PATH = "model_multi.onnx"
dynamic_axes = {'nodes': {0: 'num_nodes', 1: 'node_features'},
                'edge_index': {1: 'num_edges'},
                'output': {0: 'num_nodes'}}
torch.onnx.export(model, input_data, ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

def _worker_proc(input_q, results_q):
    session = ort.InferenceSession(ONNX_FILE_PATH, options)
    while True:
        data = input_q.get()
        if data is None:
            break
        t0 = time.time()
        output = session.run(None, {'nodes': data[0].numpy(), 'edge_index': data[1].numpy()})
        results_q.put(time.time()-t0)

input_q = Queue()
results_q = Queue()

for i in range(num_instances):
        p = Process(target=_worker_proc, args=(input_q, results_q))
        p.start()
        # pin processes to cores
        lpids = ''
        for j in range(threads_per_instance):
            lpids += str(processor_list[i*threads_per_instance + j])#FIXME:
            if j < threads_per_instance - 1:
                lpids += ','
        os.system("taskset -p -c " + lpids + " " + str(p.pid))
batches = 0

for data in data_loader:
    input_q.put((data.x, data.edge_index))
    batches += 1

for _ in range(num_instances):
    input_q.put(None)

result_tmp_q = PriorityQueue(batches)

total_time = 0
n_out = 0
while not results_q.empty():
    out = results_q.get()
    total_time+= out
    n_out+=1

print("avg time : ", total_time/n_out * 1000, " ms")

