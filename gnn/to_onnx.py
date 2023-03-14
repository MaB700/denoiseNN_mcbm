import numpy as np
import torch

from helpers import *
from helpers_custom import *
from helpers_quadrant import *

device = torch.device('cpu')
model = customGNN(graph_iters=5, hidden_size=16).to(device)
model = model.to(torch.float)
print(model)
model.load_state_dict(torch.load('model_best_5_16.pt', map_location=device))
model.eval()

x = torch.randn(20, 3)
edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                            [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]])
input_data = (x, edge_index)
ONNX_FILE_PATH = "gnn_5_16.onnx"
# dynamic_axes = {"nodes": [0, 1], "edge_index": [0, 1]}

dynamic_axes = {"nodes": {0: "num_nodes", 1:"node_features"}, "edge_index": {1: "num_edges"}, "output": {0: "num_nodes"}}
torch.onnx.export(model, input_data, ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

# load the ONNX model and do 1000 predictions and measure the time
import onnxruntime as ort
import numpy as np
import time
sess = ort.InferenceSession("gnn_5_16.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
x = np.random.randn(20, 3).astype(np.float32)
edge_index = np.array([[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                            [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]]).astype(np.int64)
start = time.time()
for i in range(1000):
    pred_ort = sess.run([output_name], {"nodes": x, "edge_index": edge_index})[0]
print(pred_ort)
print("Time: ", (time.time() - start) / 1000.0 * 1000.0, " ms")


