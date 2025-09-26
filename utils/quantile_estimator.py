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
import numpy as np


class P2QuantileEstimator:

    def __init__(self, p, initial_markers=None):
        if initial_markers is None:
            self.marker_heights = np.array([])
        else:
            self.marker_heights = np.array(initial_markers)
        self.marker_positions = np.array(range(1, 6))
        self.desired_positions = np.array(
            [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5])
        self.increments = np.array([0, p / 2, p, (1 + p) / 2, 1])

    def find_cell(self, new_observation):
        if new_observation < self.marker_heights[0]:
            return -1
        i = 0
        while i + 1 < len(self.marker_heights) and (new_observation >= self.marker_heights[i + 1]):
            i = i + 1
        return i

    def _parabolic(self, i, d):
        term1 = d / \
            float(self.marker_positions[i + 1] - self.marker_positions[i - 1])

        term2 = (self.marker_positions[i] - self.marker_positions[i - 1] + d) * \
                (self.marker_heights[i + 1] - self.marker_heights[i]) / \
            float(self.marker_positions[i + 1] - self.marker_positions[i])

        term3 = (self.marker_positions[i + 1] - self.marker_positions[i] - d) * \
                (self.marker_heights[i] - self.marker_heights[i - 1]) / \
            float(self.marker_positions[i] - self.marker_positions[i - 1])

        return self.marker_heights[i] + term1 * (term2 + term3)

    def _linear(self, i, d):
        return self.marker_heights[i] + d * \
            (self.marker_heights[i + d] - self.marker_heights[i]) / \
            float(self.marker_positions[i + d] - self.marker_positions[i])

    def update(self, new_observation):
        if len(self.marker_heights) < 5:
            self.marker_heights = np.append(
                self.marker_heights, new_observation)
            if len(self.marker_heights) == 5:
                self.marker_heights = np.sort(self.marker_heights)
            return

        i = self.find_cell(new_observation)
        if i == -1:
            self.marker_heights[0] = new_observation
            k = 0
        elif i == 4:
            self.marker_heights[4] = new_observation
            k = 3
        else:
            k = i

        self.marker_positions[k + 1:] = self.marker_positions[k + 1:] + 1
        self.desired_positions = self.desired_positions + self.increments
        self.adjust_height_values()

    def adjust_height_values(self):
        for i in range(1, 4):
            d = self.desired_positions[i] - self.marker_positions[i]
            if (d >= 1 and (self.marker_positions[i + 1] - self.marker_positions[i]) > 1) or \
                    (d <= -1 and (self.marker_positions[i - 1] - self.marker_positions[i]) < -1):

                d = -1 if d < 0 else 1

                qprime = self._parabolic(i, d)
                if self.marker_heights[i - 1] < qprime < self.marker_heights[i + 1]:
                    self.marker_heights[i] = qprime

                else:
                    qprime = self._linear(i, d)
                    self.marker_heights[i] = qprime

                self.marker_positions[i] += d

    def p_estimate(self):
        if len(self.marker_heights) > 2:
            return self.marker_heights[2]
        else:
            return np.nan

    def get_update_count(self):
        return self.desired_positions[-1]
