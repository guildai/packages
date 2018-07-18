# Notes

## Features

- MobileNet v2
  - train
  - finetune
  - finetune-all
- Quantization
- Deblurring
- Distillation
- Parameter count and model size on disk (sizing)
- Hyper parameter tuning
  - Grid search
  - Other algos
  - Multiple metrics (trade off) algos (e.g. sim to sigopt)
- TF Lite toco files
- TPU based training
- Deploy to TF serving

## Docs

For now, deploy to guild.ai/cv as a subsite with intent to move to
generalized framework docs that live here. Could link/copy/sync here
for local docs.

## `model_main.py`

The entry point for training and evaluate operations is
`object_detection/model_main.py`.

The option `--pipeline_config_path` is required and must reference a
valid pipeline protobuf config file.
