import torch
import onnx
import onnxruntime as ort
from torch_geometric.nn import SAGEConv
import os

#TODO: add dynamic axis test

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(8, 16)
        self.conv2 = SAGEConv(16, 16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = MyModel()
x = torch.randn(3, 8)
edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
expected = model(x, edge_index)
assert expected.size() == (3, 16)


dynamic_axes = {'x': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
input_data = (x, edge_index) #FIXME:
torch.onnx.export(  model, input_data, 'model.onnx',
                    input_names=('x', 'edge_index'), 
                    dynamic_axes=dynamic_axes,
                    opset_version=16)

model = onnx.load('model.onnx')
onnx.checker.check_model(model)

ort_session = ort.InferenceSession('model.onnx')

out = ort_session.run(None, {
    'x': x.numpy(),
    'edge_index': edge_index.numpy()
})[0]
out = torch.from_numpy(out)
assert torch.allclose(out, expected, atol=1e-6)
for i in range(8):
    print(expected[0,i], out[0,i])

#os.remove('model.onnx')