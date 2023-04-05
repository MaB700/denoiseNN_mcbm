import torch
import torch.nn as nn

class test(torch.nn.Module):
    def __init__(self):
        super(test, self).__init__()
        
        self.lin1 = nn.Linear(3, 10)
        self.lin2 = nn.Linear(3, 10)

        self.linx = nn.Linear(20, 1)
        self.s = nn.Sequential(
            nn.Linear(3, 10),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x1 = self.lin1(x)
        x1 = nn.Sigmoid()(x1)
        x2 = self.lin2(x)
        x2 = nn.Sigmoid()(x2)

        # x_out = self.linx(torch.cat((x1[edge_index[0]], x2[edge_index[1]]), dim=1))
        # x_out = nn.Sigmoid()(x_out)

        ex = edge_index[:, (x1[edge_index[0,:]] - x1[edge_index[1,:]]).norm(dim=1) < 0.5]
        x_out = self.linx(torch.cat((x1[ex[0]], x1[ex[1]]), dim=1))

        # find indices of abs(x1 - x2) > 0.5
        # x = x1[:, 0:3]*x1[:, 0:3] + x2[:, 0:3]*x2[:, 0:3]
        # x = (x1[:, :4] - x2[:, :4])**2
        
        # rough graph creation, loss of indices (stil known in edge_index)
        # src = x1[edge_index[0]]
        # tar = x2[edge_index[1]]
        # # radius graph creation based on embedded space
        # new_edge_index = torch.tensor([[edge_index[0, i], edge_index[0, j]] for i in range(len(src)) for j in range(len(tar)) if i != j and (src[i] - tar[j]).norm() < 0.3]).reshape(2, -1)

        return x_out
    
model = test()
num_nodes = 1000
x = torch.rand(num_nodes, 3)
t = torch.rand(num_nodes)
pos = torch.rand(num_nodes, 3)
# create edge_index tensor of size (num_edges, 2) with indices of t if t_i - t_j < 0.5 and i != j and pos_i - pos_j < 0.3
edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j and torch.abs(t[i] - t[j]) < 0.5 and (pos[i] - pos[j]).norm() < 0.3]).reshape(2, -1)
# print(edge_index)
#print(model(x, edge_index))




input_data = (x, edge_index)
# print(model(x, t))
# print(t)

# export to onnx
torch.onnx.export(model, input_data, "test.onnx", input_names=["x", "edge_index"], output_names=["output"])

import onnxruntime
import time

# load onnx model
options = onnxruntime.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.enable_profiling = False
options.execution_mode.ORT_SEQUENTIAL
sess = onnxruntime.InferenceSession("test.onnx", sess_options=options)

# run onnx model
input_name = sess.get_inputs()[0].name
input_name2 = sess.get_inputs()[1].name
output_name = sess.get_outputs()[0].name

# time 1000 runs
start = time.time()
for i in range(1000):
    pred_onnx = sess.run([output_name], {input_name: x.detach().numpy(), input_name2: edge_index.detach().numpy()})[0]
print("Time: ", (time.time() - start) / 1000.0 * 1000.0, " ms")
#print(pred_onnx)