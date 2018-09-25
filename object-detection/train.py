# Copyright 2017-2018 TensorHub, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

# Ensure matplotlib backend doesn't use tkinter
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

import _config
import _patch

log = logging.getLogger()

_patch.patch_all()

def main():
    args = _init_args()
    _validate_args(args)
    config_path = _init_config(args)
    if args.legacy:
        _legacy_train(config_path, args)
    else:
        _train(config_path, args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir", metavar="PATH", default="train",
        help="directory to write model checkpoints and logs")
    p.add_argument(
        "--train-steps", metavar="STEPS", type=int,
        help="train steps")
    p.add_argument(
        "--eval-examples", metavar="N", type=int,
        help="eval examples count")
    p.add_argument(
        "--legacy", action="store_true",
        help="use legacy train in object_detection")
    p.add_argument(
        "--clones", default=1, type=int,
        help="number of model clones")
    p.add_argument(
        "--batch-size", type=int,
        help="batch size")
    _config.add_config_args(p)
    return p.parse_args()

def _validate_args(args):
    _config.validate_config_args(args)

def _init_config(args):
    try:
        return _config.init_config(args, _args_config(args))
    except _config.ConfigError as e:
        sys.stderr.write("%s\n" % e)
        sys.exit(1)

def _args_config(args):
    config = {}
    if args.batch_size is not None:
        config["train_config"] = {"batch_size": args.batch_size}
    return config

def _legacy_train(config_path, args):
    from object_detection.legacy import train as legacy_train
    sys.argv = _legacy_train_argv(config_path, args)
    log.info("Running model_main with %s", sys.argv[1:])
    try:
        tf.app.run(legacy_train.main)
    except KeyboardInterrupt:
        sys.stderr.write("Operation stopped by user\n")

def _legacy_train_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--train_dir", args.model_dir])
    argv.extend(["--num_clones", str(args.clones)])
    return argv

def _train(config_path, args):
    from object_detection import model_main
    sys.argv = _model_main_argv(config_path, args)
    log.info("Running model_main with %s", sys.argv[1:])
    try:
        tf.app.run(model_main.main)
    except KeyboardInterrupt:
        sys.stderr.write("Operation stopped by user\n")

def _model_main_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--model_dir", args.model_dir])
    return argv

if __name__ == "__main__":
    main()
