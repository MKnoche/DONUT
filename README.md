# DONUT: A Decoder-Only Model for Trajectory Prediction

**ICCV 2025**

[arXiv](https://arxiv.org/abs/2506.06854) | [Project Page](https://vision.rwth-aachen.de/DONUT) | [YouTube](https://www.youtube.com/watch?v=6EwE89CN4Ns) | [BibTeX](#Citation)

[Markus Knoche](https://scholar.google.com/citations?user=Kx4v8IMAAAAJ)<sup>1</sup>, [Daan de Geus](https://scholar.google.com/citations?hl=de&user=4gX3HRoAAAAJ)<sup>1,2</sup>, [Bastian Leibe](https://scholar.google.com/citations?hl=de&user=ZcULDB0AAAAJ)<sup>1</sup>

<sup>1</sup> RWTH Aachen University
<sup>2</sup> Eindhoven University of Technology 

## Installation

Clone repository:

```bash
git clone https://github.com/MKnoche/DONUT.git
cd DONUT
```

Install dependencies either using uv:

```bash
pip install uv
uv sync
source .venv/bin/activate
```  

or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Adjust the root paths in `train_donut.py` according to your setup. Data will be downloaded and preprocessed automatically when running for the first time. This will take some time.

Distributed training is supported via the `devices` and `nodes` parameters. Gradient accumulation makes sure that the effective batch size is always `acc_batch_size`, as long as `batch_size * devices * nodes <= acc_batch_size`.

Training for 60 epochs on 4 NVIDIA H100 GPUs with a batch size of 8 per GPU takes about 4.5 days.

```bash
python train_donut.py
```

## Evaluation

```bash
python eval_donut.py
```

## Model Checkpoint

| Checkpoint                                                      | b-minFDE<sub>6</sub> | minFDE<sub>6</sub> | minADE<sub>6</sub> | MR<sub>6</sub> |
| --------------------------------------------------------------- | -------------------- | ------------------ | ------------------ | -------------- |
| [DONUT](https://omnomnom.vision.rwth-aachen.de/data/donut.ckpt) |                1.814 |              1.181 |              0.726 |          0.144 |

Store the checkpoint as `{args.ckpt_root}/donut/donut.ckpt`.

## Citation

If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{knoche2025donut,
  title     = {{DONUT: A Decoder-Only Model for Trajectory Prediction}},
  author    = {Knoche, Markus and de Geus, Daan and Leibe, Bastian},
  booktitle = {ICCV},
  year      = {2025}
}
```

## Acknowledgements

This project builds upon code from [QCNet](https://github.com/ZikangZhou/QCNet) (Apache-2.0 License).