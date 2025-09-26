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
import torch
import torch.nn as nn

from utils import weight_init


class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TokenMLP(nn.Module):

    def __init__(
        self,
        num_layers: str,
        aggr: str,  # cat / mean / max
        input_t: int,
        input_dim: int,
        hidden_dim: int,
        per_feature: bool = False
    ) -> None:
        super().__init__()
        self.aggr = aggr
        self.per_feature = per_feature
        if per_feature:
            input_dim, input_t = input_t, input_dim
        separate_layers = [nn.Linear(input_dim, hidden_dim)] + num_layers * [
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        self.separate_mlp = nn.Sequential(*separate_layers)

        if aggr == 'cat':
            self.proj = nn.Linear(input_t*hidden_dim, hidden_dim)
        combined_layers = num_layers * [
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        ]
        self.combined_mlp = nn.Sequential(*combined_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_feature:
            x = x.transpose(-1, -2)

        x = self.separate_mlp(x)
        if self.aggr == 'cat':
            x = x.reshape(*x.shape[:-2], -1)
            x = self.proj(x)
        elif self.aggr == 'mean':
            x = x.mean(-2)
        elif self.aggr == 'max':
            x = x.max(-2)[0]
        else:
            raise ValueError(f'aggr {self.aggr} unknown')

        x = self.combined_mlp(x)
        return x
