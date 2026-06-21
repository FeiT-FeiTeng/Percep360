# Percep360

Percep360 is a panoramic driving-scene generation framework for 360-degree perception research.

## Environment

Recommended environment:

```bash
conda create -n percep360 python=3.8 -y
conda activate percep360

pip install torch==1.10.2 torchvision==0.11.3 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements/dev.txt
pip install -r requirements/bevfusion.txt
```

Main dependencies:

```text
Python 3.8
CUDA 11.3
PyTorch 1.10.2
Diffusers 0.17.1
Accelerate 0.20.3
MMCV / MMDetection3D
```

## Configuration

The project uses Hydra-style configs under `configs/`.

Common config entries:

```text
configs/config.yaml
configs/exp/224x400.yaml
configs/runner/8gpus.yaml
configs/runner/resume.yaml
configs/dataset/Nuscenes.yaml
configs/model/SDv1.5mv_rawbox.yaml
```

Set local paths before running:

```bash
export PANODREAMER_PATH=$(pwd)
```

Place datasets and checkpoints under:

```text
data/
pretrained/
```

## Train

Four-GPU training:

```bash
accelerate launch \
  --mixed_precision fp16 \
  --gpu_ids 0,1,2,3 \
  --num_processes 4 \
  Command/train.py \
  +exp=224x400 \
  runner=8gpus
```

Resume training:

```bash
accelerate launch \
  --mixed_precision fp16 \
  --gpu_ids 0,1,2,3 \
  --num_processes 4 \
  Command/train.py \
  +exp=224x400 \
  runner=resume \
  resume_from_checkpoint=/path/to/checkpoint-xxxx
```

## Eval

Four-GPU evaluation / generation:

```bash
accelerate launch \
  --mixed_precision fp16 \
  --gpu_ids 0,1,2,3 \
  --num_processes 4 \
  perception/data_prepare/val_set_gen.py \
  resume_from_checkpoint=/path/to/weight-Exx-Sxxxxx \
  task_id=224x400 \
  fid.img_gen_dir=magicdrive-log/img_fid \
  +fid=data_gen \
  +exp=224x400
```

Single-process evaluation wrapper:

```bash
PERCEP360_CHECKPOINT=/path/to/weight-Exx-Sxxxxx \
PERCEP360_FID_DIR=magicdrive-log/img_fid \
python val_main.py
```

## Citation

```bibtex
@article{teng2025hallucinating,
  title={Hallucinating 360 {$\backslash$deg}: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting},
  author={Teng, Fei and Luo, Kai and Wu, Sheng and Li, Siyu and Guo, Pujun and Wei, Jiale and Peng, Kunyu and Zhang, Jiaming and Yang, Kailun},
  journal={arXiv preprint arXiv:2507.06971},
  year={2025}
}
```
