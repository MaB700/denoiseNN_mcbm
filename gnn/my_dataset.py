import os
import shutil
import numpy as np
from typing import Optional
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.data.datapipes import functional_transform
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform

class MyDataset(InMemoryDataset):
    def __init__(self, root, dataset = "train", reload=False, undirected=True):
        self.dataset = dataset
        self.undirected = undirected
        if reload:            
            path = os.path.join(root, 'processed/', self.dataset)
            if os.path.exists(path):
                shutil.rmtree(path)            
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["E:/ML_data/test/x.npy", 
                "E:/ML_data/test/y.npy",
                "E:/ML_data/test/pos.npy",
                "E:/ML_data/test/edge_index.npy"]
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed/train') if self.dataset=="train" else os.path.join(self.root, 'processed/test')
    
    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        pass

    def process(self):
        x = np.load(self.raw_paths[0])
        y = np.load(self.raw_paths[1])
        pos = np.load(self.raw_paths[2])
        edge_index = np.load(self.raw_paths[3])

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        pos = torch.from_numpy(pos).float()
        edge_index = torch.from_numpy(edge_index).int()

        data_list = [Data(  x=x[i],
                            y=y[i],
                            pos=pos[i])
                            for i in range(len(y))]
        
        for data in data_list: #TODO: are those deep copys? 
            data = T.RadiusGraph(0.5, max_num_neighbors=4)(data)
            data = self.rm_edges(data) # remove edges outside abs time difference [workaround]
            data = T.ToUndirected()(data)
            data = T.Distance()(data)
            data = TimeDifference()(data) # might consider sign
            data.pos = None
            # data.t = None

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def rm_edges(self, data, dim=2):
        src_time = torch.index_select(data.x[:, dim], 0, data.edge_index[0,:])
        target_time = torch.index_select(data.x[:, dim], 0, data.edge_index[1,:])
        data.edge_index = data.edge_index[:, torch.abs(src_time - target_time) < 0.1]
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
        (row, col), time, pseudo = data.edge_index, data.x[:, 2], data.edge_attr #FIXME: -> t

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
    