import numpy as np
import os
import torch
import torch.nn as nn

def make_mlp(input_size, output_size, hidden_size, num_layers,
             hidden_activation="ReLU", output_activation="ReLU",
             layer_norm=False, dropout=None):
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

        layers.append(hidden_activation())
        
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))    
        
    # Final layer
    if num_layers == 1:
        layers.append(nn.Linear(input_size, output_size))
    else:
        layers.append(nn.Linear(hidden_size, output_size))
    
        layers.append(output_activation())
        
    if output_activation is not None and output_size != 1:
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        if layer_norm:
            layers.append(nn.LayerNorm(output_size))
    
    return nn.Sequential(*layers)

def scatter_add_attention(encoded_nodes, encoded_edges, edge_list):
    start, end = edge_list[0], edge_list[1]

    src = encoded_nodes[end]*encoded_edges
    index = start.unsqueeze(-1)
    add_messages = torch.zeros(encoded_nodes.shape, dtype=src.dtype, device=encoded_nodes.device).scatter_add_(0, index.repeat((1,src.shape[1])), src)
    denom = torch.zeros(encoded_nodes.shape, dtype=src.dtype, device=encoded_nodes.device).scatter_add_(0, index.repeat((1,src.shape[1])), torch.ones_like(src))
    mean_messages = add_messages/denom
    
    aggr_nodes = torch.cat([add_messages, mean_messages], dim=1) # + out_messages
    # aggr_nodes = add_messages # mean_messages
    return aggr_nodes

class customGNN(nn.Module):
    def __init__(self, graph_iters = 3, hidden_size = 16, num_layers = 3,
                 dropout = None, layer_norm = False):
        super(customGNN, self).__init__()

        self.graph_iters = graph_iters
        # Setup node encoder
        self.node_encoder = make_mlp(
            input_size=3, 
            output_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            output_activation=None,
            dropout=dropout,
            layer_norm=layer_norm
        )

        # The edge network computes edge scores from connected nodes
        self.edge_network = nn.ModuleList(make_mlp(
            input_size=2*(hidden_size) + 2,
            output_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm=layer_norm
        ) for _ in range(self.graph_iters))

        # The node network computes new node features
        self.node_network = nn.ModuleList(make_mlp(
            input_size=3*(hidden_size),
            output_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm=layer_norm
        ) for _ in range(self.graph_iters - 1))

        self.node_out_network = make_mlp(
            input_size=3*(hidden_size),
            output_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_activation="Sigmoid",
            dropout=dropout,
            layer_norm=layer_norm
        )


    def forward(self, x, edge_index, edge_attr):
        # x_in = x
        x = self.node_encoder(x)

        start, end = edge_index[0], edge_index[1]

        # Loop over iterations of edge and node networks
        for i in range(self.graph_iters):
            # Previous hidden state
            x0 = x # for skip connection
            # x = torch.cat([x, x_in], dim=1)

            # Compute new edge score
            # edge_inputs = torch.cat([x[start], x[end]], dim=1)
            edge_inputs = torch.cat([x[start], x[end], edge_attr], dim=1)
            e = self.edge_network[i](edge_inputs)
            e = torch.sigmoid(e)

            weighted_messages = scatter_add_attention(x, e, edge_index)

            node_inputs = torch.cat([x, weighted_messages], dim=1)

            if i == self.graph_iters - 1:
                x = self.node_out_network(node_inputs)
            else:
                x = self.node_network[i](node_inputs)                
                # Residual connection
                # x = torch.cat([x, x_in], dim=-1)
                x = x + x0
        
        return x