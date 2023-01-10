import numpy as np
import os
import torch
import torch.nn as nn

def make_mlp(
    input_size,
    output_size,
    hidden_size,
    num_layers,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    # Hidden layers
    for i in range(num_layers - 1):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(hidden_activation())   
        
    # Final layer
    if num_layers == 1:
        layers.append(nn.Linear(input_size, output_size))
    else:
        layers.append(nn.Linear(hidden_size, output_size))
    
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(output_size))
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

class customGNN(nn.Module):
    def __init__(self, graph_iters = 3):
        super(customGNN, self).__init__()

        self.graph_iters = graph_iters
        # Setup node encoder
        self.node_encoder = make_mlp(
            input_size=3, 
            output_size=32,
            hidden_size=32,
            num_layers=1,
            output_activation=None
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = nn.ModuleList(make_mlp(
            input_size=2*(32+3),
            output_size=1,
            hidden_size=32,
            num_layers=3
        ) for _ in range(self.graph_iters))

        # The node network computes new node features
        self.node_network = nn.ModuleList(make_mlp(
            input_size=2*(32+3),
            output_size=32,
            hidden_size=32,
            num_layers=3,
        ) for _ in range(self.graph_iters - 1))

        self.node_out_network = make_mlp(
            input_size=2*(32+3),
            output_size=1,
            hidden_size=32,
            num_layers=3,
            output_activation="Sigmoid"
        )


    def forward(self, x, edge_index): #, data
        # Encode the graph features into the hidden space
        input_x = x
        x = self.node_encoder(x) # [num_nodes, 3] -> [num_nodes, hidden]
        x = torch.cat([x, input_x], dim=-1) # 

        start, end = edge_index[0], edge_index[1]

        # Loop over iterations of edge and node networks
        for i in range(self.graph_iters):
            # Previous hidden state
            x0 = x # for skip connection

            # Compute new edge score
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = self.edge_network[i](edge_inputs)
            e = torch.sigmoid(e)

            # Sum weighted node features coming into each node
            #             weighted_messages_in = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])
            #             weighted_messages_out = scatter_add(e * x[end], start, dim=0, dim_size=x.shape[0])

            weighted_messages = scatter_add_attention(x, e, edge_index)

            # Compute new node features
            #             node_inputs = torch.cat([x, weighted_messages_in, weighted_messages_out], dim=1)
            node_inputs = torch.cat([x, weighted_messages], dim=1)
            if i == self.graph_iters - 1:
                x = self.node_out_network(node_inputs)
            else:
                x = self.node_network[i](node_inputs)                
                # Residual connection
                x = torch.cat([x, input_x], dim=-1)
                x = x + x0
        
        return x