# Copyright (c) 2025, Markus Knoche. All rights reserved.
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
import torch
import torch.nn as nn
from torch_cluster import radius
from torch_geometric.utils import dense_to_sparse

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle
from utils import bipartite_dense_to_sparse


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        rel_dim = 4

        self.rel_emb = FourierEmbedding(
            input_dim=rel_dim, hidden_dim=hidden_dim)

        self.attn_layer = AttentionLayer(
            hidden_dim=hidden_dim, bipartite=True, has_pos_emb=True)

        self.apply(weight_init)

    def forward(self, src_x, src_pos, src_head, edges, dst_x=None, dst_pos=None, dst_head=None):

        num_agents, num_modes, num_hist, num_channels = src_x.shape

        src_pos = src_pos.reshape(-1, 2)
        src_head = src_head.reshape(-1)
        if dst_x is None:
            dst_x, dst_pos, dst_head = src_x, src_pos, src_head
            dst_same = True
        else:
            if dst_x.shape[-2] != 1:
                raise ValueError(
                    f'x_dst must contain exactly one time step, but has {dst_x.shape[-2]}: {dst_x.shape}')
            dst_pos = dst_pos.reshape(-1, 2)
            dst_head = dst_head.reshape(-1)
            dst_same = False
        if num_modes != 1:
            # make single-mode edges multi-modal
            num_curr = num_hist if dst_same else 1
            rep_edges = edges.repeat_interleave(num_modes, dim=1)
            rep_range = torch.arange(
                num_modes, device=edges.device).repeat(edges.shape[1])
            edges = torch.stack([
                rep_edges[0] % num_hist + num_modes*num_hist *
                (rep_edges[0]//num_hist) + num_hist*rep_range,
                rep_edges[1] % num_curr + num_modes*num_curr *
                (rep_edges[1]//num_curr) + num_curr*rep_range,
            ])

        dst_head_vec = torch.stack([dst_head.cos(), dst_head.sin()], dim=-1)

        rel_pos = src_pos[edges[0]] - dst_pos[edges[1]]
        dist = torch.linalg.norm(rel_pos, dim=-1)
        direction = angle_between_2d_vectors(
            ctr_vector=dst_head_vec[edges[1]], nbr_vector=rel_pos)
        rel_head = wrap_angle(src_head[edges[0]] - dst_head[edges[1]])

        src_t = torch.arange(src_x.shape[2], device=src_x.device)[
            None, None].repeat(*src_x.shape[:2], 1).flatten()
        dst_t = torch.arange(dst_x.shape[2], device=dst_x.device)[
            None, None].repeat(*dst_x.shape[:2], 1).flatten()
        if not dst_same:
            dst_t += num_hist
        time_rel = src_t[edges[0]] - dst_t[edges[1]]

        rel = torch.stack([
            dist,
            direction,
            rel_head,
            time_rel
        ], dim=-1)
        rel = self.rel_emb(rel)

        src_x = src_x.reshape(-1, num_channels)
        dst_x = dst_x.reshape(-1, num_channels)

        x = self.attn_layer((src_x, dst_x), rel, edges)
        x = x.reshape(num_agents, num_modes,
                      num_hist if dst_same else 1, num_channels)
        return x

    @staticmethod
    def create_edges(mask_src, mask_dst=None):

        if mask_dst is None:
            mask_dst = mask_src
            dst_same = True
        else:
            if mask_dst.shape[-1] != 1:
                raise ValueError(
                    f'dst_mask must contain exactly one time step, but has {mask_dst.shape[-1]}: {mask_dst.shape}')
            dst_same = False

        mask_src = mask_src[:, 0]
        mask_dst = mask_dst[:, 0]

        edges = bipartite_dense_to_sparse(
            mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        if dst_same:
            edges = edges[:, edges[1] > edges[0]]

        return edges


class RoadAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        pl_rel_dim = 3

        self.pl_rel_emb = FourierEmbedding(
            input_dim=pl_rel_dim, hidden_dim=hidden_dim)

        self.attn_layer = AttentionLayer(
            hidden_dim=hidden_dim, bipartite=True, has_pos_emb=True
        )

        self.apply(weight_init)

    def forward(self, map_x, edges, pl_rel, dst_x):
        num_agents, num_modes, num_steps, num_channels = dst_x.shape

        dst_x = dst_x.reshape(-1, num_channels)

        rel = self.pl_rel_emb(pl_rel)

        x = self.attn_layer((map_x, dst_x), rel, edges)
        x = x.reshape(num_agents, num_modes, num_steps, num_channels)

        return x

    @staticmethod
    def create_edges(pl_pos, pl_head, pl_batch, dst_pos, dst_head, dst_mask, dst_batch, radius_r2a, max_degree=100):

        num_modes = dst_pos.shape[1]
        num_steps = dst_pos.shape[2]

        dst_batch = dst_batch[:, None].repeat(
            1, num_modes*num_steps).reshape(-1)
        dst_pos = dst_pos.reshape(-1, 2)
        dst_mask = dst_mask.reshape(-1)
        dst_head = dst_head.reshape(-1)

        dst_head_vec = torch.stack([dst_head.cos(), dst_head.sin()], dim=-1)

        # pl edges
        pl_edges = radius(x=dst_pos, y=pl_pos, r=radius_r2a,
                          batch_x=dst_batch, batch_y=pl_batch, max_num_neighbors=max_degree)

        pl_edges = pl_edges[:, dst_mask[pl_edges[1]]]

        # pl rel
        rel_pos = pl_pos[pl_edges[0]] - dst_pos[pl_edges[1]]
        dist = torch.linalg.norm(rel_pos, dim=-1)
        direction = angle_between_2d_vectors(
            ctr_vector=dst_head_vec[pl_edges[1]], nbr_vector=rel_pos)
        rel_head = wrap_angle(pl_head[pl_edges[0]] - dst_head[pl_edges[1]])

        pl_rel = torch.stack([
            dist,
            direction,
            rel_head
        ], dim=-1)

        return pl_edges, pl_rel


class SocialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        rel_dim = 3

        self.rel_emb = FourierEmbedding(
            input_dim=rel_dim, hidden_dim=hidden_dim)

        self.attn_layer = AttentionLayer(
            hidden_dim=hidden_dim, bipartite=False, has_pos_emb=True)

        self.apply(weight_init)

    def forward(self, x, pos, head, edges):
        num_agents, num_modes, num_steps, num_channels = x.shape

        # order must be TxMxBA for batch to be increasing only
        x = x.permute(2, 1, 0, 3).reshape(-1, num_channels)
        pos = pos.permute(2, 1, 0, 3).reshape(-1, 2)
        head = head.permute(2, 1, 0).reshape(-1)

        dst_head_vec = torch.stack([head.cos(), head.sin()], dim=-1)

        rel_pos = pos[edges[0]] - pos[edges[1]]
        dist = torch.linalg.norm(rel_pos, dim=-1)
        direction = angle_between_2d_vectors(
            ctr_vector=dst_head_vec[edges[1]], nbr_vector=rel_pos)
        rel_head = wrap_angle(head[edges[0]] - head[edges[1]])

        rel = torch.stack([
            dist,
            direction,
            rel_head
        ], dim=-1)
        rel = self.rel_emb(rel)

        x = self.attn_layer(x, rel, edges)
        x = x.reshape(num_steps, num_modes, num_agents,
                      num_channels).permute(2, 1, 0, 3)

        return x

    @staticmethod
    def create_edges(pos, mask, batch, radius_a2a):
        _, num_modes, num_steps, _ = pos.shape
        batch_size = torch.max(batch)+1

        # order must be TxMxBA for batch to be increasing only
        batch = batch[None, None, :].repeat(num_steps, num_modes, 1)
        batch += torch.arange(num_modes,
                              device=batch.device)[None, :, None] * batch_size
        batch += torch.arange(num_steps, device=batch.device)[
            :, None, None] * num_modes * batch_size

        pos = pos.permute(2, 1, 0, 3).reshape(-1, 2)
        batch = batch.reshape(-1)
        mask = mask.permute(2, 1, 0).reshape(-1)

        edges = radius(x=pos, y=pos, r=radius_a2a, batch_x=batch,
                       batch_y=batch, max_num_neighbors=500)
        edges = edges[:, mask[edges[0]] & mask[edges[1]]]

        return edges


class ModeAttention(nn.Module):
    def __init__(self, hidden_dim, num_modes, pred_steps):
        super().__init__()

        rel_dim = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.time_emb = nn.Embedding(pred_steps+1, hidden_dim)

        self.rel_emb = FourierEmbedding(
            input_dim=rel_dim, hidden_dim=hidden_dim)

        self.attn_layer = AttentionLayer(
            hidden_dim=hidden_dim, bipartite=False, has_pos_emb=True)

        self.apply(weight_init)

    def forward(self, x, pos, head, edges, pred_step):
        num_agents, num_modes, num_steps, num_channels = x.shape

        if num_modes > 1:
            rep_edges = num_modes * \
                edges.repeat_interleave(num_modes**2, dim=1)

            edges = torch.stack([
                rep_edges[0] + torch.arange(num_modes, device=edges.device).repeat(
                    num_modes).repeat(edges.shape[1]),
                rep_edges[1] + torch.arange(num_modes, device=edges.device).repeat_interleave(
                    num_modes).repeat(edges.shape[1]),
            ])

        # mode embedding
        if num_modes > 1:
            x = x + self.mode_emb.weight[None, :, None]

        # time embedding
        x = x + self.time_emb(
            torch.tensor(pred_step, device=x.device, dtype=torch.long))

        # order must be BAxTxM for batch to be increasing only
        x = x.permute(0, 2, 1, 3).reshape(-1, num_channels)
        pos = pos.permute(0, 2, 1, 3).reshape(-1, 2)
        head = head.permute(0, 2, 1).reshape(-1)

        dst_head_vec = torch.stack([head.cos(), head.sin()], dim=-1)

        rel_pos = pos[edges[0]] - pos[edges[1]]
        dist = torch.linalg.norm(rel_pos, dim=-1)
        direction = angle_between_2d_vectors(
            ctr_vector=dst_head_vec[edges[1]], nbr_vector=rel_pos)
        rel_head = wrap_angle(head[edges[0]] - head[edges[1]])

        rel = torch.stack([
            dist,
            direction,
            rel_head
        ], dim=-1)
        rel = self.rel_emb(rel)

        x = self.attn_layer(x, rel, edges)
        x = x.reshape(num_agents, num_steps, num_modes,
                      num_channels).permute(0, 2, 1, 3)

        return x

    @staticmethod
    def create_edges(mask):
        mask = mask[:, 0]
        num_agents, num_steps = mask.shape

        # order must be BAxT(xM) for batch to be increasing only
        batch = torch.arange(num_agents*num_steps, device=mask.device)

        batch = batch.reshape(-1)
        mask = mask.reshape(-1)

        edges = dense_to_sparse(
            batch[:, None] == batch[None, :])[0]

        edges = edges[:, mask[edges[0]] & mask[edges[1]]]

        return edges
