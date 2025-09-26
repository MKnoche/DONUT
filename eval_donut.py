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
from argparse import ArgumentParser
import pytorch_lightning as pl
from datamodules import ArgoverseV2DataModule
from predictors import Donut
from pathlib import Path
import torch


def load_args():

    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='donut')
    parser.add_argument('--data_root', type=str, default='data/av2')
    parser.add_argument('--ckpt_root', type=str, default='ckpts')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=15)
    Donut.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(0)

    args = load_args()

    model = Donut(**vars(args))
    datamodule = ArgoverseV2DataModule(**vars(args))

    ckpt_root = Path(args.ckpt_root) / args.name

    # get newest ckpt
    if ckpt_root.exists():
        ckpt_paths = list(
            sorted(
                ckpt_root.glob('*'),
                key=lambda x: x.lstat().st_mtime
            ))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['state_dict'])
            print(f'Loaded checkpoint {ckpt_path}.')
        else:
            raise FileNotFoundError(
                f'Checkpoint directory {ckpt_root} is empty.')
    else:
        raise FileNotFoundError(
            f'Checkpoint directory {ckpt_root} does not exist.')

    trainer = pl.Trainer()
    trainer.validate(model, datamodule)
