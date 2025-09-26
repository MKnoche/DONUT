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

from layers import TemporalAttention, RoadAttention, SocialAttention, ModeAttention
from layers import Tokenizer, Detokenizer
from utils import P2QuantileEstimator
from utils import global_to_local, local_to_global
from torch.nn.functional import elu


class DonutLayer(nn.Module):
    def __init__(
            self, hidden_dim, t_per_tok, type_count, over_predict,
            refine, has_feature_input, has_feature_output, has_prob_output,
            order, repetitions, radius_r, radius_s, edge_limit, mode_config):
        super().__init__()
        self.edge_limit = edge_limit
        if any([c not in 'trsm' for c in order]):
            raise ValueError(f'only t, r, s and m are valid, not', order)
        self.refine = refine
        self.order = order
        self.repetitions = repetitions
        self.radius_r = radius_r
        self.radius_s = radius_s

        self.tokenizer = Tokenizer(
            hidden_dim, t_per_tok, type_count, has_feature_input)

        attentions = []
        for _ in range(self.repetitions):
            for c in self.order:
                if c == 't':
                    attentions.append(TemporalAttention(
                        hidden_dim))
                elif c == 'r':
                    attentions.append(RoadAttention(
                        hidden_dim))
                elif c == 's':
                    attentions.append(SocialAttention(
                        hidden_dim))
                elif c == 'm':
                    attentions.append(ModeAttention(
                        hidden_dim, **mode_config))
        self.attentions = nn.ModuleList(attentions)

        self.detokenizer = Detokenizer(
            hidden_dim, t_per_tok, over_predict, has_feature_output, has_prob_output)

        self.p2qe = P2QuantileEstimator(self.edge_limit)

    def forward(
        self, x_pos, x_head, x_type, x_mask, x_batch, pred_step,
        pl_x, pl_pos, pl_head, pl_batch,
        proposed=None, past=None, feature_input=None
    ):
        if past is not None:
            past_temps, past_pos, past_head, past_mask = past
            if x_pos.shape[1] != past_pos.shape[1]:
                num_modes = x_pos.shape[1]
                past_temps = [
                    temp.repeat_interleave(num_modes, 1)
                    for temp in past_temps
                ]
                past_pos = past_pos.repeat_interleave(num_modes, 1)
                past_head = past_head.repeat_interleave(num_modes, 1)
                past_mask = past_mask.repeat_interleave(num_modes, 1)

        tok_pos_m = x_pos[:, :, :, -1:]
        tok_head_m = x_head[:, :, :, -1:]
        x_pos, x_head = global_to_local(x_pos, x_head, tok_pos_m, tok_head_m)
        x = self.tokenizer(
            x_pos, x_head, x_type, feature_input)
        tok_pos = tok_pos_m[:, :, :, 0]
        tok_head = tok_head_m[:, :, :, 0]
        tok_mask = x_mask.all(3)

        if 't' in self.order:
            if past is None:
                edges_t = TemporalAttention.create_edges(tok_mask)
            else:
                edges_t = TemporalAttention.create_edges(past_mask, tok_mask)
        if 'r' in self.order:
            edges_l, pl_rel = RoadAttention.create_edges(
                pl_pos, pl_head, pl_batch,
                tok_pos, tok_head, tok_mask, x_batch,
                self.radius_r)
        if 's' in self.order:
            edges_s = SocialAttention.create_edges(
                tok_pos, tok_mask, x_batch, self.radius_s)
        if 'm' in self.order:
            edges_m = ModeAttention.create_edges(tok_mask)

        if self.training and self.edge_limit != 1:
            max_edges = max(
                edges_t.shape[1],
                edges_l.shape[1],
                edges_s.shape[1],
                edges_m.shape[1],
            )
            self.p2qe.update(max_edges)
            if self.p2qe.get_update_count() > min(10 / (1-self.edge_limit), 1000):
                max_edge_count = int(self.p2qe.p_estimate())
                if edges_t.shape[1] > max_edge_count:
                    edges_t = edges_t[:, :max_edge_count]
                if edges_l.shape[1] > max_edge_count:
                    edges_l = edges_l[:, :max_edge_count]
                    pl_rel = pl_rel[:max_edge_count]
                if edges_s.shape[1] > max_edge_count:
                    edges_s = edges_s[:, :max_edge_count]
                if edges_m.shape[1] > max_edge_count:
                    edges_m = edges_m[:, :max_edge_count]

        temps = []
        attn_i = 0
        t_i = 0
        for _ in range(self.repetitions):
            for c in self.order:
                if c == 't':
                    temps.append(x.clone())
                    if past is None:
                        x = self.attentions[attn_i](
                            x, tok_pos, tok_head, edges_t)
                    else:
                        x = self.attentions[attn_i](past_temps[t_i], past_pos,
                                                    past_head, edges_t, x, tok_pos, tok_head)
                    t_i += 1
                elif c == 'r':
                    x = self.attentions[attn_i](
                        pl_x, edges_l, pl_rel, x)
                elif c == 's':
                    x = self.attentions[attn_i](x, tok_pos, tok_head, edges_s)
                elif c == 'm':
                    x = self.attentions[attn_i](x, tok_pos, tok_head, edges_m,
                                                pred_step=pred_step)
                attn_i += 1

        if past is None:
            past = (temps, tok_pos, tok_head, tok_mask)
        else:
            past = (
                [torch.cat([past_temp, temp], dim=-2)
                 for past_temp, temp in zip(past_temps, temps)],
                torch.cat([past_pos, tok_pos], dim=-2),
                torch.cat([past_head, tok_head], dim=-1),
                torch.cat([past_mask, tok_mask], dim=-1),
            )

        xs = self.detokenizer(x)
        if self.refine:
            prop_pos, prop_head = global_to_local(
                *proposed, tok_pos_m, tok_head_m)
            xs['pos'] = xs['pos'] + prop_pos.detach().reshape(xs['pos'].shape)
            xs['head'] = xs['head'] + \
                prop_head.detach().reshape(xs['head'].shape)
        else:
            xs['pos'] = torch.cumsum(xs['pos'], dim=3)
            xs['head'] = torch.cumsum(0.3*torch.tanh(xs['head']), dim=3)

        xs['scale'] = 1 + elu(xs['scale'])
        xs['conc'] = 1 + elu(xs['conc'])

        xs['pos'], xs['head'] = local_to_global(
            xs['pos'], xs['head'], tok_pos_m, tok_head_m)

        return xs, past
