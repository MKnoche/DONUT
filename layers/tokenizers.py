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

from layers import FourierEmbedding
from layers import TokenMLP, MLPLayer
from utils import angle_between_2d_vectors


class Tokenizer(nn.Module):
    def __init__(self, hidden_dim, t_per_tok, type_count, has_feature_input):
        super().__init__()
        inp_dim = 8
        self.fourier_emb = FourierEmbedding(
            inp_dim, hidden_dim
        )
        self.token_model = TokenMLP(
            num_layers=3,
            aggr='cat',
            input_t=t_per_tok-1,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            per_feature=False
        )

        self.x_mlp = MLPLayer(hidden_dim, hidden_dim, hidden_dim)
        if has_feature_input:
            self.feature_mlp = MLPLayer(hidden_dim, hidden_dim, hidden_dim)
        self.type_emb = nn.Embedding(type_count, hidden_dim)
        self.joint_mlp = MLPLayer(hidden_dim, hidden_dim, hidden_dim)

    def forward(
        self, x_pos, x_head, x_type, feature_input=None
    ):

        x_mot_vec = x_pos[..., 1:, :] - x_pos[..., :-1, :]
        x_vel = torch.linalg.norm(x_mot_vec, dim=-1)
        x_head_vec = torch.stack([x_head.cos(), x_head.sin()], dim=-1)
        head_vs_mot = angle_between_2d_vectors(
            ctr_vector=x_head_vec[..., :-1, :], nbr_vector=x_mot_vec)
        x_head_rel = x_head[..., 1:] - x_head[..., :-1]

        x = torch.cat([
            x_pos[..., :-1, :],      # position token-relative
            x_head[..., :-1, None],  # heading token-relative
            x_mot_vec,               # position token- and time-relative
            x_head_rel[..., None],   # heading token- and time-relative
            x_vel[..., None],        # velocity
            head_vs_mot[..., None],  # difference between heading and motion
        ], dim=-1)
        x = self.fourier_emb(
            x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)
        x = self.token_model(x)

        x = self.x_mlp(x)
        if feature_input is not None:
            x += self.feature_mlp(feature_input)
        x += self.type_emb(x_type)
        x = self.joint_mlp(x)

        return x


class Detokenizer(nn.Module):
    def __init__(self, hidden_dim, t_per_tok, over_predict, has_feature_output, has_prob_output):
        super().__init__()
        self.feat_len = (1 + over_predict) * t_per_tok
        named_output_dims = {
            'pos': 2 * self.feat_len,
            'head': self.feat_len,
            'scale': 2 * self.feat_len,
            'conc': self.feat_len,
        }
        if has_feature_output:
            named_output_dims['feats'] = hidden_dim
        if has_prob_output:
            named_output_dims['prob'] = hidden_dim

        self.joint_mlp = MLPLayer(hidden_dim, hidden_dim, hidden_dim)
        self.out_mlps = nn.ModuleDict({
            name: MLPLayer(hidden_dim, hidden_dim, output_dim)
            for name, output_dim in named_output_dims.items()
        })

    def forward(self, x):

        x = self.joint_mlp(x)
        xs = {name: mlp(x) for name, mlp in self.out_mlps.items()}

        num_agents, num_modes = x.shape[:2]
        xs['pos'] = xs['pos'].reshape(
            num_agents, num_modes, -1, self.feat_len, 2)
        xs['head'] = xs['head'].reshape(
            num_agents, num_modes, -1, self.feat_len)
        xs['scale'] = xs['scale'].reshape(
            num_agents, num_modes, -1, self.feat_len, 2)
        xs['conc'] = xs['conc'].reshape(
            num_agents, num_modes, -1, self.feat_len)
        return xs
