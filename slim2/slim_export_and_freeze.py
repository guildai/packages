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
import glob
import os
import re
import sys

import tensorflow as tf

from tensorflow.python.tools import freeze_graph

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
    args, rest_args = _init_args()
    _export_graph(args, rest_args)
    _freeze_graph(args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_checkpoint", metavar="PATH",
        required=True,
        help=(
            "path checkpoint used in export; may be a directory "
            "or a ckpt path"))
    p.add_argument(
        "--output_dir", metavar="DIR",
        default=".",
        help="directory to write exported graph files")
    p.add_argument(
        "--output_node_names", metavar="VAL",
        help="output node names")
    return p.parse_known_args()

def _export_graph(cmd_args, rest_args):
    # export_inference_graph ignores args to main so we need to hack
    # sys.argv prior to importing module.
    with _util.argv(_export_argv(cmd_args, rest_args)):
        import export_inference_graph
        try:
            tf.app.run(export_inference_graph.main)
        except SystemExit as e:
            if e.code:
                raise

def _export_argv(cmd_args, rest_args):
    return ["export_inference_graph"] + rest_args + [
        "--alsologtostderr",
        "--dataset_name", "custom",
        "--output_file", _graph_pb(cmd_args),
    ]

def _graph_pb(args):
    return os.path.join(args.output_dir, "graph.pb")

def _freeze_graph(cmd_args):
    freeze_args = [
        "--input_graph", _graph_pb(cmd_args),
        "--input_binary",
        "--input_checkpoint", _latest_checkpoint(cmd_args.input_checkpoint),
        "--output_graph", _frozen_graph_pb(cmd_args),
        "--output_node_names", cmd_args.output_node_names,
    ]
    # freeze_graph hard-codes use of sys.argv
    argv_save = sys.argv
    sys.argv = ["freeze_graph"] + freeze_args
    try:
        freeze_graph.run_main()
    except SystemExit as e:
        if e.code:
            raise
    finally:
        sys.argv = argv_save

def _frozen_graph_pb(args):
    return os.path.join(args.output_dir, "frozen_graph.pb")

def _latest_checkpoint(path):
    if os.path.isdir(path):
        checkpoint = tf.train.latest_checkpoint(path)
    elif path.endswith(".ckpt"):
        checkpoint = _latest_checkpoint_for_base(path)
    else:
        checkpoint = path
    if not checkpoint:
        _util.error("cannot find latest checkpoint for path: %s" % path)
    return checkpoint

def _latest_checkpoint_for_base(base):
    max = 0
    latest = None
    for match in glob.glob("%s-*.meta" % base):
        match_step = _checkpoint_step(match)
        if match_step > max:
            max = match_step
            latest = match
    return os.path.splitext(latest)[0]

def _checkpoint_step(meta_path):
    m = re.search(r"-([0-9]+)\.meta$", meta_path)
    assert m, meta_path
    return int(m.group(1))

if __name__ == "__main__":
    main(sys.argv)
