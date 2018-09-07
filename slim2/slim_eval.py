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

import sys

import tensorflow as tf

sys.path.insert(0, "slim")

from datasets import dataset_factory

import _custom_dataset
import _util

def main(argv):
    if "--dataset_name" in argv:
        _util.error("--dataset_name is not supported")
    if "--model_name" not in argv:
        _util.error("--model_name is required")
    dataset_factory.datasets_map = {
        "custom": _custom_dataset
    }
    _eval(argv)

def _eval(argv):
    with _util.argv(_eval_args(argv)):
        import eval_image_classifier
        try:
            tf.app.run(eval_image_classifier.main)
        except SystemExit as e:
            if e.code:
                raise

def _eval_args(argv):
    return argv + ["--dataset_name", "custom"]

if __name__ == "__main__":
    main(sys.argv)
