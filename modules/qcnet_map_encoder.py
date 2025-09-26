# Copyright (c) 2023, Zikang Zhou. All rights reserved.
# Modified by Markus Knoche, 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle


class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_layers: int) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers

        input_dim_x_pt = 1
        input_dim_x_pl = 0
        input_dim_r_pt2pl = 3
        input_dim_r_pl2pl = 3

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(3, hidden_dim)
        self.type_pl_emb = nn.Embedding(4, hidden_dim)
        self.int_pl_emb = nn.Embedding(3, hidden_dim)

        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(
            input_dim=input_dim_x_pt, hidden_dim=hidden_dim)
        self.x_pl_emb = FourierEmbedding(
            input_dim=input_dim_x_pl, hidden_dim=hidden_dim)
        self.r_pt2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim)
        self.r_pl2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, bipartite=True,
                            has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, bipartite=False,
                            has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data['map_point']['position'][:, :2].contiguous()
        orient_pt = data['map_point']['orientation'].contiguous()
        pos_pl = data['map_polygon']['position'][:, :2].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        orient_vector_pl = torch.stack(
            [orient_pl.cos(), orient_pl.sin()], dim=-1)

        x_pt = data['map_point']['magnitude'].unsqueeze(-1)
        x_pl = None
        x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()),
                                 self.side_pt_emb(data['map_point']['side'].long())]
        x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),
                                 self.int_pl_emb(data['map_polygon']['is_intersection'].long())]
        x_pt = self.x_pt_emb(continuous_inputs=x_pt,
                             categorical_embs=x_pt_categorical_embs)
        x_pl = self.x_pl_emb(continuous_inputs=x_pl,
                             categorical_embs=x_pl_categorical_embs)

        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - \
            pos_pl[edge_index_pt2pl[1]]
        rel_orient_pt2pl = wrap_angle(
            orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])
        r_pt2pl = torch.stack(
            [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                         nbr_vector=rel_pos_pt2pl[:, :2]),
                rel_orient_pt2pl], dim=-1)
        r_pt2pl = self.r_pt2pl_emb(
            continuous_inputs=r_pt2pl, categorical_embs=None)

        edge_index_pl2pl = data['map_polygon',
                                'to', 'map_polygon']['edge_index']
        edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                               batch=data['map_polygon']['batch'] if isinstance(
                                                   data, Batch) else None,
                                               loop=False, max_num_neighbors=300)
        type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type']
        type_pl2pl_radius = type_pl2pl.new_zeros(
            edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
        edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
                                                   edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - \
            pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(
            orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
        r_pl2pl = torch.stack(
            [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                         nbr_vector=rel_pos_pl2pl[:, :2]),
                rel_orient_pl2pl], dim=-1)
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[
                                   self.type_pl2pl_emb(type_pl2pl.long())])

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i](
                (x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim)

        return {'x_pt': x_pt, 'x_pl': x_pl}
