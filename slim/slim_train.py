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

import yaml

import tensorflow as tf

sys.path.insert(0, "slim")

import _custom_dataset
import _util

log = logging.getLogger("slim_train")

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
        "--auto-scale",
        default="yes",
        help="Whether or not to adjust flags on multi-GPU systems (yes)")
    p.add_argument(
        "--num_clones",
        type=int,
        default=1,
        help="Number of clones to deploy model to.")
    p.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate.")
    p.add_argument(
        "--num_readers",
        type=int,
        default=4,
        help="Number of data readers.")
    p.add_argument(
        "--num_preprocessing_threads",
        type=int,
        default=4,
        help="Number of preprocessing threads.")
    return p.parse_known_args(argv)

def _train(cmd_args, rest_args):
    # train_image_classifier ignores args to main so we need to hack
    # sys.argv prior to importing module.
    train_argv = _train_argv(cmd_args, rest_args)
    _log_train_argv(train_argv)
    with _util.argv(train_argv):
        import train_image_classifier
        try:
            tf.app.run(train_image_classifier.main)
        except SystemExit as e:
            if e.code:
                raise

def _train_argv(cmd_args, rest_args):
    auto_scale = _auto_scale(cmd_args)
    argv = list(rest_args)
    argv.extend(_num_clones_args(cmd_args, auto_scale))
    argv.extend(_learning_rate_args(cmd_args, auto_scale))
    argv.extend(_num_readers_args(cmd_args, auto_scale))
    argv.extend(_num_preprocessing_threads_args(cmd_args, auto_scale))
    argv.extend(["--dataset_name", "custom"])
    return argv

def _auto_scale(cmd_args):
    return bool(yaml.safe_load(cmd_args.auto_scale))

def _num_clones_args(cmd_args, auto_scale):
    if auto_scale:
        clones = max(_gpu_count(), 1)
        log.info("auto scaled clones: %i", clones)
    else:
        clones = cmd_args.num_clones
    return ["--num_clones", str(clones)]

def _gpu_count():
    from tensorflow.python.client import device_lib
    return sum([
        dev.device_type == "GPU"
        for dev in device_lib.list_local_devices()])

def _learning_rate_args(cmd_args, auto_scale):
    learning_rate = cmd_args.learning_rate
    if auto_scale:
        learning_rate = max(_gpu_count(), 1) * learning_rate
        log.info("auto scaled learning rate: %f", learning_rate)
    return ["--learning_rate", str(learning_rate)]

def _num_readers_args(cmd_args, auto_scale):
    if auto_scale:
        readers = _cpu_count() // 2
    else:
        readers = cmd_args.num_readers
    return ["--num_readers", str(readers)]

def _cpu_count():
    import psutil
    return psutil.cpu_count(logical=True)

def _num_preprocessing_threads_args(cmd_args, auto_scale):
    if auto_scale:
        threads = _cpu_count() // 2
    else:
        threads = cmd_args.num_preprocessing_threads
    return ["--num_preprocessing_threads", str(threads)]

def _log_train_argv(argv):
    log.info("train_image_classifier args: %r", argv)
    formatted = _format_args(argv)
    _try_write_guild_attr("_train_image_classifier_args", formatted)

def _format_args(args):
    lines = []
    cur = []
    for arg in args + ["--"]:
        if arg.startswith("--"):
            if cur:
                lines.append(" ".join(cur))
                cur = []
        cur.append(arg)
    return "\n".join(lines) + "\n"

def _try_write_guild_attr(name, val_str):
    attrs_dir = os.path.join(".guild", "attrs")
    if os.path.exists(attrs_dir):
        attr_path = os.path.join(attrs_dir, name)
        with open(attr_path, "w") as f:
            f.write(val_str)

if __name__ == "__main__":
    main(sys.argv)
