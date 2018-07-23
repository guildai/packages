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

The option `--model_dir` may be used to specify where checkpoints and
logs are written. If this isn't specified `model_main` creates a
unique diretory under `/tmp`

## Pipeline config

The configuration used by object_detection is quite complex: it's
represented by the protobuf specs under `object_detection/protos`.

We want to support flag based configuration and so will have to
generate configuration files (protobuf text format). Here's an
approach that should work:

- Provide default config for each model - we can use the config
  provided under `object_detection/samples/config`. These are good
  starting points that we can modify as needed.

- The model operations that use `model_main` must be associated with a
  default config name. This will tell Guild what to use as a starting
  point.

- The Guild wrapper `train.py` will support a generalized *flag*
  interface that will support specifying config values. We can use a
  dot notation to traverse the config attribute hierarchy. E.g. a flag
  `model.ssd.num_classes` could be used to modify the ssd model's
  `num_classes` attribute.

- Models in the Guild file can use flags and main script args to
  modify the default config.

Example:

``` yaml
- model: pets-mobilenet
  extends: ssd-detector
  operations:
    train:
      main:
        train --config ssd_mobilenet_v1_pets
              --model.ssd.num_classes 37
      flags:
        train-steps:
          arg-name: train_config.num_steps
          default: 50000
```

- Config that we control such as dataset and checkpoint file location
  will be set by Guild wrappers.

- Once the configuration is modified, Guild will save it to a local
  file to be used as the operation config via the
  `--pipeline_config_path` option.

One point of consideration is whether we should maintain our own
default config or always reference whatever Google provides under
`object_detection/samples/config`.

Maintaining our own is probably the right decision as it will insulate
our Guild file and user Guild files from Google's upstream changes.

Another approach might be to maintain configuration at more granular
levels (e.g. separate model config from training, etc.). This would
allow more sensible mixing and matching of configuration and minimize
the proliferation of configuration files, which we're starting to see
under `object_detection/samples/config`.

Update: On further inspection, it's probably a bad idea to attempt
factoring the config into smaller, reusable parts:

- While there are indeed sharable pieces, they tend to have
  exceptions, which will require a config finetuning strategy. The
  result may end up being less understandable and maintainable than
  monlithic configuration.

- The exceptions across config may include config "drift", where
  copy-and-pasted settings weren't updated consistently across config
  files. The temptation will be to fix these by sharing configuration
  speculatively. As we don't have the understanding needed to make
  these changes, we run the risk of breaking working models in the
  interest of "cleaning up" the configuration.

Our strategy must at this time preserve Google's configuration. The
easiest way to do that is to use it directly -- i.e. use the files
under `samples/config` as they are.
