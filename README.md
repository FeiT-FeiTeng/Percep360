<div align="center">

# Hallucinating 360°: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting

Fei Teng · Kai Luo · Wu Sheng · Siyu Li · Jiale Wei ·
[Jiaming Zhang](https://www.researchgate.net/profile/Jiaming-Zhang-10) ·
[Kunyu Peng](https://www.researchgate.net/profile/Kunyu-Peng) ·
[Kailun Yang](https://www.researchgate.net/profile/Kailun-Yang)

[Paper](https://arxiv.org/abs/2507.06971)

</div>

<div align="center">
  <img src="111.png" width="820" height="400" />
</div>

## Update

- 2025.07.21 Init repository.
- 2025.07.21 Release the [arXiv](https://arxiv.org/abs/2507.06971) version.
- 2026.06.21 The integration of code is released.

## Abstract

Panoramic perception holds significant potential for autonomous driving, enabling vehicles to acquire a comprehensive 360° surround view in a single shot. However, autonomous driving is a data-driven task. Complete panoramic data acquisition requires complex sampling systems and annotation pipelines, which are time-consuming and labor-intensive. Although existing street view generation models have demonstrated strong data regeneration capabilities, they can only learn from the fixed data distribution of existing datasets and cannot achieve high-quality, controllable panoramic generation.

In this paper, we propose the first panoramic generation method Percep360 for autonomous driving. Percep360 enables coherent generation of panoramic data with control signals based on the stitched panoramic data. Percep360 focuses on two key aspects: coherence and controllability. Specifically, to overcome the inherent information loss caused by the pinhole sampling process, we propose the Local Scenes Diffusion Method (LSDM). LSDM reformulates the panorama generation as a spatially continuous diffusion process, bridging the gaps between different data distributions. Additionally, to achieve controllable panoramic image generation, we propose a Probabilistic Prompting Method (PPM). PPM dynamically selects the most relevant control cues, enabling controllable panoramic image generation.

We evaluate the effectiveness of the generated images from three perspectives: image quality assessment, controllability, and their utility in real-world Bird's Eye View (BEV) segmentation. The generated data consistently outperforms the original stitched images in no-reference quality metrics and enhances downstream perception models.

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

## Data Preparation

Please follow the data preparation protocol from [OneBEV](https://github.com/JialeWei/OneBEV/tree/main).

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
