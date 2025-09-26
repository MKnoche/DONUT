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
from utils.quantile_estimator import P2QuantileEstimator
from utils.coordinate_transformations import global_to_local, local_to_global
from utils.geometry import angle_between_2d_vectors
from utils.geometry import side_to_directed_lineseg
from utils.geometry import wrap_angle
from utils.graph import bipartite_dense_to_sparse
from utils.graph import merge_edges
from utils.list import safe_list_index
from utils.weight_init import weight_init
