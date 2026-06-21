# Legacy Command Notes

These commands are kept for reference from the original local experiments. Replace all placeholder paths before running them.

## Accelerate Setup

```bash
accelerate config
```

Typical local setup:

```text
this machine
Multi-GPU
4 processes
FP16
```

## Training

```bash
accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 4 \
  Command/train.py +exp=224x400 runner=8gpus
```

Resume training:

```bash
accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 4 \
  Command/train.py +exp=224x400 runner=resume \
  resume_from_checkpoint=/path/to/checkpoint-xxxx
```

## Generate Images For Metrics

```bash
python perception/data_prepare/val_set_gen.py \
  resume_from_checkpoint=/path/to/weight-Exx-Sxxxxx \
  task_id=224x400 \
  fid.img_gen_dir=magicdrive-log/img_fid \
  +fid=data_gen \
  +exp=224x400
```

Accelerate alternative:

```bash
accelerate launch perception/data_prepare/val_set_gen.py \
  resume_from_checkpoint=/path/to/weight-Exx-Sxxxxx \
  task_id=224x400 \
  fid.img_gen_dir=magicdrive-log/img_fid \
  +fid=data_gen \
  +exp=224x400
```

## Saved Model Formats

- `checkpoint-xxxx`: Accelerate state used to resume training.
- `weight-Exx-Sxxxxx`: Diffusers model directory used for inference and evaluation.

If you use a local FID Inception weight, set:

```bash
export FID_INCEPTION_WEIGHTS=/path/to/pt_inception-2015-12-05-6726825d.pth
```
