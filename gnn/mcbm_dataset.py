import os
import shutil
import numpy as np
import uproot
from typing import Optional
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.data.datapipes import functional_transform
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform

class MyDataset(InMemoryDataset):
    def __init__(self, root=None, dataset="train", N=1000, radius = 7.0, max_num_neighbors = 8, reload=False, undirected=True):
        self.N = N
        self.dataset = dataset
        self.undirected = undirected
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        file_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.dirname(__file__) + "/../data/")
        if reload:
            path = os.path.abspath(file_dir + "/../data/" + 'processed/' + self.dataset)
            if os.path.exists(path):
                shutil.rmtree(path)            
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'data.root') if self.dataset=="train" else os.path.join(self.root, 'data_test.root')]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed/train') if self.dataset=="train" else os.path.join(self.root, 'processed/test')
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        pass

    def process(self):
        f = uproot.open(self.raw_paths[0])
        time = np.array(f['train']['time'].array(entry_stop=self.N))
        tar = np.array(f['train']['tar'].array(entry_stop=self.N))

        data_list = [self.initial_graph(i, time, tar) for i in range(len(time))]
        
        for data in data_list: #TODO: are those deep copys? 
            data = T.RadiusGraph(self.radius, max_num_neighbors=self.max_num_neighbors)(data)
            data = self.rm_edges(data) # remove edges outside abs time difference [workaround]
            data = T.ToUndirected()(data)
            data = T.Distance()(data)
            data = TimeDifference()(data) # might consider sign
            data.pos = None
            data.t = None

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def initial_graph(self, index, time, tar):
        n = np.count_nonzero(time[index, :] > 0.01)
        hit_indices = np.nonzero(time[index, :] > 0.01)
        hits = time[index, :][hit_indices]
        y_pos, x_pos = np.divmod(hit_indices, 32)
        
        # Pos
        pos = np.zeros((n, 2))
        pos[:, 0] = (x_pos.astype('float32')) # x_coord [0.0, 31.0]cm
        pos[:, 1] = (y_pos.astype('float32')) # y_coord [0.0, 71.0]cm
        # time
        t = np.zeros((n, 1))
        t[:, 0] = hits.astype('float32')*26.0 # time [1.0,26.0]ns
        
        # Nodes
        x = np.zeros((n, 3))
        x[:, 0] = hits.astype('float32') # time [0,1]
        x[:, 1] = pos[:, 0]/31.0 # x_coord [0,1]
        x[:, 2] = pos[:, 1]/71.0 # y_coord [0,1]
        
        # Labels
        y = np.zeros((n, 1))
        y[:, 0] = tar[index, :][hit_indices]

        return Data(x=torch.from_numpy(x).float(),
                    y=torch.from_numpy(y).float(),
                    pos=torch.from_numpy(pos).float(),
                    t=torch.from_numpy(t).float())

    def rm_edges(self, data, delta_t = 5.0):
        src_time = torch.index_select(data.t, 0, data.edge_index[0,:]).view(-1)
        target_time = torch.index_select(data.t, 0, data.edge_index[1,:]).view(-1)
        data.edge_index = data.edge_index[:, torch.abs(src_time - target_time) <= delta_t]
        return data

@functional_transform('timedifference')
class TimeDifference(BaseTransform):
    r"""Saves the relative time difference of linked nodes in its edge attributes
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, norm: bool = True, max_value: Optional[float] = None,
                 cat: bool = True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        (row, col), time, pseudo = data.edge_index, data.t, data.edge_attr

        dist = torch.abs(time[col] - time[row]).view(-1, 1)
        # dist = (time[col] - time[row]).view(-1, 1)
        
        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
    