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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger
from datamodules import ArgoverseV2DataModule
from predictors import Donut
from pathlib import Path
import torch


def load_args():

    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='donut')
    parser.add_argument('--data_root', type=str, default='data/av2')
    parser.add_argument('--ckpt_root', type=str, default='ckpts')
    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=60)
    Donut.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = load_args()

    model = Donut(**vars(args))
    datamodule = ArgoverseV2DataModule(**vars(args))

    ckpt_root = Path(args.ckpt_root) / args.name
    checkpointer = ModelCheckpoint(Path(args.ckpt_root) / args.name,
                                   monitor='train_loss', save_top_k=3, mode='min')
    # auto resume
    ckpt_path = None
    if ckpt_root.exists():
        # get newest ckpt
        ckpt_paths = list(
            sorted(
                ckpt_root.glob('*'),
                key=lambda x: x.lstat().st_mtime
            ))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]

    logger = CSVLogger(Path(args.log_root), args.name)

    # compute batch accumulation
    total_batch_size = (args.batch_size * args.devices * args.nodes)
    if total_batch_size > args.acc_batch_size:
        args.acc_batch_size = total_batch_size
        print(f'Set batch size to {total_batch_size}')
    assert args.acc_batch_size % total_batch_size == 0
    acc_batches = args.acc_batch_size // total_batch_size

    trainer = pl.Trainer(devices=args.devices, num_nodes=args.nodes,
                         strategy=DDPStrategy(gradient_as_bucket_view=True),
                         callbacks=[checkpointer],
                         accumulate_grad_batches=acc_batches,
                         logger=logger,
                         max_epochs=args.max_epochs)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
