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
import sys

import yaml

import tensorflow as tf

sys.path.insert(0, "slim")

import _custom_dataset
import _util

def main(argv):
    if "--dataset_name" in argv:
        _util.error("--dataset_name is not supported")
    if "--model_name" not in argv:
        _util.error("--model_name is required")
    cmd_args, rest_args = _init_args(argv)
    _custom_dataset.patch_dataset_factory()
    _train(cmd_args, rest_args)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--num_clones",
        type=int,
        help="Number of clones to deploy model to.")
    p.add_argument(
        "--optimize-defaults",
        default="yes",
        help="Whether or not to provide optimized default values (yes)")
    return p.parse_known_args(argv)

def _train(cmd_args, rest_args):
    # train_image_classifier ignores args to main so we need to hack
    # sys.argv prior to importing module.
    with _util.argv(_train_argv(cmd_args, rest_args)):
        import train_image_classifier
        try:
            tf.app.run(train_image_classifier.main)
        except SystemExit as e:
            if e.code:
                raise

def _train_argv(cmd_args, rest_args):
    optimize = _use_optimized_defaults(cmd_args)
    rest_args += _num_clones_args(cmd_args, optimize)
    return rest_args + [
        "--dataset_name", "custom"
    ]

def _use_optimized_defaults(cmd_args):
    return bool(yaml.safe_loads(cmd_args.optimize_defaults))

def _num_clones_args(cmd_args, optimize):
    if cmd_args.num_clones is not None:
        return ["--num_clones", str(cmd_args.num_clones)]
    elif optimize:
        gpus = _gpu_count()
        if gpus > 0:
            return ["--num_clones", str(gpus)]
    return []


def _gpu_count():
    from tensorflow.python.client import device_lib
    return sum([
        dev.device_type == "GPU"
        for dev in device_lib.list_local_devices()])

if __name__ == "__main__":
    main(sys.argv)
