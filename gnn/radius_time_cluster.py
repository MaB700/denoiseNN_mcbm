import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Optional

import torch
try:
    import torch_cluster
except ImportError:
    torch_cluster = None

@functional_transform('radius_time_graph')
class RadiusTimeGraph(BaseTransform):
    def __init__(
        self,
        r: float,
        t_diff: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ):
        self.r = r
        self.t_diff = t_diff
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def __call__(self, data: Data) -> Data:
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None

        data.edge_index = radius_time_graph(
            data.pos,
            data.t,
            self.r,
            self.t_diff,
            batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'

def radius_time_graph(x: Tensor, t: Tensor, r: float, t_diff: float, batch: OptTensor = None,
                 loop: bool = False, max_num_neighbors: int = 32,
                 flow: str = 'source_to_target',
                 num_workers: int = 1) -> Tensor:
    
    return torch_cluster.radius_graph(x, t, r, t_diff, batch, loop, max_num_neighbors,
                                      flow, num_workers)

@torch.jit.script
def radius_graph(x: torch.Tensor, t: torch.Tensor, r: float, t_diff: float,
                 batch: Optional[torch.Tensor] = None, loop: bool = False,
                 max_num_neighbors: int = 32, flow: str = 'source_to_target',
                 num_workers: int = 1) -> torch.Tensor:

    assert flow in ['source_to_target', 'target_to_source']
    # t_mask = torch.abs(t[:, None] - t[None, :]) < t_diff # FIXME:
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors if loop else max_num_neighbors + 1,
                        num_workers)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)

@torch.jit.script
def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None, max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    batch_size = 1
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1
    if batch_y is not None:
        assert y.size(0) == batch_y.numel()
        batch_size = max(batch_size, int(batch_y.max()) + 1)

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None
    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers)