import torch
import onnx
import onnxruntime as ort
from torch_geometric.nn import SAGEConv, GATConv, ARMAConv
import os

#TODO: add dynamic axis test

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ARMAConv(8, 16)
        self.conv2 = ARMAConv(16, 16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = MyModel()
x = torch.randn(3, 8, requires_grad=True)
x2 = torch.randn(5, 8)
edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], requires_grad=False)
edge_index2 = torch.tensor([[0, 1, 2, 3, 4, 3], 
                            [1, 0, 2, 1, 0, 4]])
expected = model(x, edge_index)
grads = torch.autograd.grad(expected, 
                            inputs=x, 
                            grad_outputs=torch.ones_like(expected, requires_grad=True),
                            retain_graph=True)
print(grads)



# assert expected.size() == (3, 16)

# expected2 = model(x2, edge_index2)
# assert expected2.size() == (5, 16)

# # dynamic_axes = {'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
# dynamic_axes = {"x": [0, 1], "edge_index": [0, 1], "y": [0, 1]}
# input_data = (x, edge_index) #FIXME:
# torch.onnx.export(  model, input_data, 'model.onnx',
#                     input_names=['x', 'edge_index'],
#                     output_names=['y'], 
#                     dynamic_axes=dynamic_axes,
#                     opset_version=16)

# model = onnx.load('model.onnx')
# onnx.checker.check_model(model)

# ort_session = ort.InferenceSession('model.onnx')

# out = ort_session.run(None, {
#     'x': x.numpy(),
#     'edge_index': edge_index.numpy()
# })[0]
# out = torch.from_numpy(out)
# assert torch.allclose(out, expected, atol=1e-6)
# for i in range(8):
#     print(expected[0,i], out[0,i])

# out2 = ort_session.run(None, {
#     'x': x2.numpy(),
#     'edge_index': edge_index2.numpy()
# })[0]
# out2 = torch.from_numpy(out2)
# assert torch.allclose(out2, expected2, atol=1e-6)
# for i in range(8):
#     print(expected2[0,i], out2[0,i])

#os.remove('model.onnx')
