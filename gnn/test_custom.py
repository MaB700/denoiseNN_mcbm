import time

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import remove_isolated_nodes
import torch.nn as nn

import onnxruntime as ort

from helpers_custom import *
device = "cuda" if torch.cuda.is_available() else "cpu"

# print pytorch version
print(torch.__version__)
# model to .onnx works properly with pytorch 1.13

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)

def scatter_add_attention(encoded_nodes, encoded_edges, edge_list):
    start, end = edge_list[0], edge_list[1]

    src = encoded_nodes[end]*encoded_edges
    index = start.unsqueeze(-1)
    in_messages = torch.zeros(encoded_nodes.shape, dtype=src.dtype, device=encoded_nodes.device).scatter_add(0, index.repeat((1,src.shape[1])), src) 

    src = encoded_nodes[start]*encoded_edges
    index = end.unsqueeze(-1)
    out_messages = torch.zeros(encoded_nodes.shape, dtype=src.dtype, device=encoded_nodes.device).scatter_add(0, index.repeat((1,src.shape[1])), src) 
    
    aggr_nodes = in_messages + out_messages
    
    return aggr_nodes

class ResAGNN(nn.Module):
    def __init__(self, hparams):
        super(ResAGNN, self).__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        
        self.hparams = hparams
        
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"]],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * (hparams["in_channels"] + hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            (hparams["in_channels"] + hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        # Encode the graph features into the hidden space
        input_x = x
        x = self.node_encoder(x) # [num_nodes, 3] -> [num_nodes, hidden]
        x = torch.cat([x, input_x], dim=-1) # 

        start, end = edge_index[0], edge_index[1]

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            # Previous hidden state
            x0 = x # for skip connection

            # Compute new edge score
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = self.edge_network(edge_inputs)
            e = torch.sigmoid(e)

            # Sum weighted node features coming into each node
            #             weighted_messages_in = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])
            #             weighted_messages_out = scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])

            weighted_messages = scatter_add_attention(x, e, edge_index)

            # Compute new node features
            #             node_inputs = torch.cat([x, weighted_messages_in, weighted_messages_out], dim=1)
            node_inputs = torch.cat([x, weighted_messages], dim=1)
            x = self.node_network(node_inputs)

            # Residual connection
            x = torch.cat([x, input_x], dim=-1)
            x = x + x0

        # Compute final edge scores; use original edge directions only
        clf_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.edge_network(clf_inputs).squeeze(-1)

hparams = torch.load("hyper_parameters.ckpt")
hparams.update({"n_graph_iters": 3})
hparams.update({"layernorm": False})
hparams.update({"nb_edge_layer": 2})
hparams.update({"nb_node_layer": 2})
hparams.update({"hidden": 32})
# print content of .ckpt file
# print(hparams)
#model = ResAGNN(hparams)
model = customGNN()
# print(model)

x = torch.randn(3, 3)
x2 = torch.randn(10, 3)
x3 = torch.randn(35, 3)
edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
edge_index2 = torch.tensor([[1, 0, 2, 2, 2, 4, 3, 7, 6], 
                            [0, 1, 3, 4, 7, 0, 4, 8, 9]])
edge_index3 = torch.tensor([[0, 1, 2, 3, 4, 3, 7, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                            [1, 0, 2, 1, 0, 4, 8, 9, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28]])

n_disconnected_graphs = 32
data = [Data(x=x3, edge_index=edge_index3) for _ in range(n_disconnected_graphs)]
data_loader = DataLoader(data, batch_size=n_disconnected_graphs)
datax = None
for data in data_loader:
    datax = data

input_data = (x, edge_index)
input_data2 = (x2, edge_index2)
input_data3 = (x3, edge_index3)
input_datax = (datax.x, datax.edge_index)
ONNX_FILE_PATH = "ResAGNN_model.onnx"
dynamic_axes = {"nodes": [0, 1], "edge_index": [0, 1]}
torch.onnx.export(model, input_data, ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

expected = model(*input_data)
expected2 = model(*input_data2)
expected3 = model(*input_data3)
expectedx = model(*input_datax)
# print(expected3)
options = ort.SessionOptions()
options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4 # athena uses 1
options.inter_op_num_threads = 4
# options.enable_profiling=True
options.execution_mode.ORT_PARALLEL
session = ort.InferenceSession(ONNX_FILE_PATH, None)
out = session.run(None, {"nodes": input_data[0].numpy(), "edge_index": input_data[1].numpy()})[0]
out2 = session.run(None, {"nodes": input_data2[0].numpy(), "edge_index": input_data2[1].numpy()})[0]
out3 = session.run(None, {"nodes": input_data3[0].numpy(), "edge_index": input_data3[1].numpy()})[0]
outx = session.run(None, {"nodes": input_datax[0].numpy(), "edge_index": input_datax[1].numpy()})[0]

import time
start = time.time()
for i in range(1000):
    session.run(None, {"nodes": input_data3[0].numpy(), "edge_index": input_data3[1].numpy()})[0]
end = time.time()
print("Average time of inference: ", (end - start) / 1000 * 1000, "ms")

start = time.time()
for i in range(1000):
    session.run(None, {"nodes": input_datax[0].numpy(), "edge_index": input_datax[1].numpy()})[0]
end = time.time()
print("Average time of inference x: ", (end - start) / 1000 * 1000, "ms")





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

