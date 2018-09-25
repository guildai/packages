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

import tensorflow as tf

from object_detection import export_inference_graph

from gpkg.slim import _util

import _config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def main():
    args = _parse_args()
    _validate_args(args)
    config_path = _init_config(args)
    sys.argv = _export_argv(config_path, args)
    log.info("Running export_inference_graph with %s", sys.argv[1:])
    tf.app.run(export_inference_graph.main)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", metavar="PATH", required=True,
        help="path checkpoint used to freeze graph")
    p.add_argument(
        "--checkpoint-step", metavar="N", type=int,
        help="checkpoint step to use in freeze (defaults to latest)")
    p.add_argument(
        "--output-dir", metavar="PATH", default="graph",
        help="directory to write graphs")
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

def _export_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--input_type", "image_tensor"])
    argv.extend(["--trained_checkpoint_prefix", _checkpoint_prefix(args)])
    argv.extend(["--output_directory", args.output_dir])
    return argv

def _checkpoint_prefix(args):
    return _util.input_checkpoint(args.checkpoint, args.checkpoint_step)

if __name__ == "__main__":
    main()
