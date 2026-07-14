'''
File to define Neural Networks
'''

import torch_cluster
from torch_geometric.transforms import RadiusGraph
from torch_geometric.nn import radius_graph
from torch.nn import PairwiseDistance
import torch
from torch import nn
from torch_scatter import scatter
from torch_sparse import SparseTensor
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from torch.cuda.amp import autocast

T = TypeVar('T', bound='Module')

from MachineLearning.GNN_Layers import *

torch.backends.cudnn.benchmark = True


class GNN_Grapher:

    def __init__(self,radius,max_num_neighbors) -> None:
        self._gnn_grapher = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_num_neighbors)

    def build_gnn_graph(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atomic_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN_Grapher_2(GNN_Grapher):

    def build_gnn_graph(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atom_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class _GNN_fix_cuda:

    _lock_device = False

    def to(self, *args, **kwargs):
        if self._lock_device:
            pass
        else:
            super().to(*args, **kwargs)

class GNN_delta(GNN_Grapher_2):

    def __init__(self,fraction=0.5,radius=0.6, max_num_neighbors=32, device=None, jittable=False, hidden=128):
        # Initiate Graph Builder
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        self._nobatch = False

        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._grapher = RadiusGraph(r=self._radius, loop=False, max_num_neighbors=self._max_num_neighbors)

        # Init Distance Calculation
        self._distancer = PairwiseDistance()
        self._jittable = jittable


        self._gnn_radius = radius
        GNN_Grapher_2.__init__(self,radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(3 , hidden,radius,device,hidden).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(hidden , hidden,radius,device,hidden).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(hidden , 2,radius,device,hidden).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(3 , hidden,radius,device,hidden)
            self.interaction2 = IN_layer_all_swish_2pass(hidden, hidden,radius,device,hidden)
            self.interaction3 = IN_layer_all_swish_2pass(hidden, 2,radius,device,hidden)

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph

        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atom_features

        # Do message passing

        # ADD small correction
        Bcn = self.interaction1(edge_index=gnn_edge_index,x=x,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        energies = self.interaction3(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)

        # Evaluate GB energies


        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients
        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1, 1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return forces

