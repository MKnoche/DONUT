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
from typing import Optional, Tuple

import torch
from torch_scatter import gather_csr
from torch_scatter import segment_csr


def topk(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:
                inds_topk = torch.topk(
                    prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(
                pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(
                pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones(
                (pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk


def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    outs = []
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    outs.append(pred)
    target = target[filter_mask]
    outs.append(target)
    if batch is not None:
        batch = batch[filter_mask]
        outs.append(batch)
    if prob is not None:
        prob = prob[filter_mask]
        outs.append(prob)
    valid_mask = valid_mask[filter_mask]
    outs.append(valid_mask)
    return tuple(outs)
