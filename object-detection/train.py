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
import os
import sys

# Ensure proper encoding of floats in protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Ensure matplotlib backend doesn't use tkinter
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

from object_detection import model_main

import _config
import _patch

log = logging.getLogger()

def main():
    args = _init_args()
    _validate_args(args)
    config_path = _init_config(args)
    if os.getenv("SKIP_TRAIN") == "1":
        log.info("SKIP_TRAIN set, skipping train")
        return
    sys.argv = _model_main_argv(config_path, args)
    log.info("Running model_main with %s", sys.argv[1:])
    _patch.patch_all()
    try:
        tf.app.run(model_main.main)
    except KeyboardInterrupt:
        sys.stderr.write("Operation stopped by user\n")

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
    _config.add_config_args(p)
    return p.parse_args()

def _validate_args(args):
    _config.validate_config_args(args)

def _init_config(args):
    try:
        return _config.init_config(args)
    except _config.ConfigError as e:
        sys.stderr.write("%s\n" % e)
        sys.exit(1)

def _model_main_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--model_dir", args.model_dir])
    return argv

if __name__ == "__main__":
    main()
