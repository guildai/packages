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
    _eval(cmd_args, rest_args)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_path", metavar="PATH",
        help="checkpoint path")
    p.add_argument(
        "--checkpoint_step", metavar="STEP",
        type=int,
        help="checkpoint step to evaluate")
    return p.parse_known_args(argv)

def _eval(cmd_args, rest_args):
    with _util.argv(_eval_args(cmd_args, rest_args)):
        import eval_image_classifier
        try:
            tf.app.run(eval_image_classifier.main)
        except SystemExit as e:
            if e.code:
                raise

def _eval_args(cmd_args, rest_args):
    return rest_args + [
        "--checkpoint_path", _checkpoint_path(cmd_args),
        "--dataset_name", "custom"
    ]

def _checkpoint_path(args):
    return _util.input_checkpoint(args.checkpoint_path, args.checkpoint_step)

if __name__ == "__main__":
    main(sys.argv)
