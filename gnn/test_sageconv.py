import torch
import onnxruntime as ort
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, ARMAConv, DynamicEdgeConv
import time
from helpers_quadrant import *
# print pytorch version
print(torch.__version__)
#TODO: add dynamic axis test

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(3, 16, project=True, aggr="add")
        self.conv2 = SAGEConv(16, 16, project=True, aggr="add")
        self.conv3 = SAGEConv(16, 1, project=True, aggr="add")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index).sigmoid()
    


data = CreateGraphDataset_quadrant('../data/data.root:train', 10000, 7)
# data = mcbm_dataset.MyDataset(  dataset="train", N = 16, reload=True, \
#                                 radius = 7, max_num_neighbors = 8)
loader = DataLoader(data, batch_size=1)

model = MyModel()
x = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(5, 3)
edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], requires_grad=False)
edge_index2 = torch.tensor([[0, 1, 2, 3, 4, 3], 
                            [1, 0, 2, 1, 0, 4]])
expected = model(data[0].x, data[0].edge_index)
# grads = torch.autograd.grad(expected, 
#                             inputs=x, 
#                             grad_outputs=torch.ones_like(expected, requires_grad=True),
#                             retain_graph=True)
# print(grads)



# assert expected.size() == (3, 16)

expected2 = model(x2, edge_index2)
# assert expected2.size() == (5, 16)

# # dynamic_axes = {'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
dynamic_axes = {"x": [0, 1], "edge_index": [0, 1], "y": [0, 1]}
input_data = (data[0].x, data[0].edge_index)
torch.onnx.export(  model, input_data, 'arma.onnx',
                    input_names=['x', 'edge_index'],
                    output_names=['y'], 
                    dynamic_axes=dynamic_axes,
                    opset_version=16)

# model = onnx.load('model.onnx')
# onnx.checker.check_model(model)

options = ort.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.enable_profiling = False
options.execution_mode.ORT_SEQUENTIAL

ort_session = ort.InferenceSession('arma.onnx', sess_options=options)

out = ort_session.run(None, {
    'x': input_data[0].detach().numpy(),
    'edge_index': input_data[1].detach().numpy()
})[0]
out = torch.from_numpy(out)
assert torch.allclose(out, expected, atol=1e-6)
for i in range(len(expected)):
    print(expected[i,0], out[i,0])

x_list = []
edge_index_list = []
for i in range(len(data)):
    x_list.append(data[i].x.detach().numpy())
    edge_index_list.append(data[i].edge_index.detach().numpy())

# time for 1000 runs
start = time.time()
for i in range(len(data)):
    out = ort_session.run(None, {'x': x_list[i], 'edge_index': edge_index_list[i]})[0]
print((time.time() - start) * 1000 / len(data))

# out2 = ort_session.run(None, {
#     'x': x2.numpy(),
#     'edge_index': edge_index2.numpy()
# })[0]
# out2 = torch.from_numpy(out2)
# assert torch.allclose(out2, expected2, atol=1e-6)
# for i in range(8):
#     print(expected2[0,i], out2[0,i])

#os.remove('model.onnx')
