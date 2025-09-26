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
from utils.geometry import wrap_angle


def global_to_local(x_pos, x_head, anchor_pos, anchor_head):

    assert x_pos.shape[:-1] == x_head.shape
    assert anchor_pos.shape[:-1] == anchor_head.shape

    x_pos = x_pos - anchor_pos

    cos, sin = anchor_head.cos(), anchor_head.sin()
    rot_mat = torch.zeros(*anchor_head.shape, 2, 2, device=x_pos.device)
    rot_mat[..., 0, 0] = cos
    rot_mat[..., 0, 1] = -sin
    rot_mat[..., 1, 0] = sin
    rot_mat[..., 1, 1] = cos
    if rot_mat.shape[-3] == 1:
        rot_mat = rot_mat.repeat_interleave(x_pos.shape[-2], dim=-3)
    rot_mat = rot_mat.to(x_pos.dtype)
    x_pos = torch.einsum('...i,...ij->...j', x_pos, rot_mat)

    x_head = x_head - anchor_head
    x_head = wrap_angle(x_head)

    return x_pos, x_head


def local_to_global(x_pos, x_head, anchor_pos, anchor_head):

    assert x_pos.shape[:-1] == x_head.shape
    assert anchor_pos.shape[:-1] == anchor_head.shape

    cos, sin = anchor_head.cos(), anchor_head.sin()
    rot_mat = torch.zeros(*anchor_head.shape, 2, 2, device=x_pos.device)
    rot_mat[..., 0, 0] = cos
    rot_mat[..., 0, 1] = sin
    rot_mat[..., 1, 0] = -sin
    rot_mat[..., 1, 1] = cos
    if rot_mat.shape[-3] == 1:
        rot_mat = rot_mat.repeat_interleave(x_pos.shape[-2], dim=-3)
    rot_mat = rot_mat.to(x_pos.dtype)
    x_pos = torch.einsum('...i,...ij->...j', x_pos, rot_mat)

    x_pos = x_pos + anchor_pos

    x_head = x_head + anchor_head
    x_head = wrap_angle(x_head)

    return x_pos, x_head
