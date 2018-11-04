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

import tensorflow as tf

from gpkg.slim import _util

import _config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def main():
    args = _parse_args()
    _validate_args(args)
    config_path = _init_config(args)
    if args.export_type == "default":
        _export_default(config_path, args)
    elif args.export_type == "tflite-ssd":
        _export_tflite_ssd(config_path, args)
    else:
        assert False, args.export_type

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
    p.add_argument(
        "--export-type", default="default", choices=("default", "tflite-ssd"),
        help="type of graph export (default, tflite-ssd)")
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

def _export_default(config_path, args):
    from object_detection import export_inference_graph
    sys.argv = _export_default_argv(config_path, args)
    log.info("Running export_inference_graph with %s", sys.argv[1:])
    tf.app.run(export_inference_graph.main)

def _export_default_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--input_type", "image_tensor"])
    argv.extend(["--trained_checkpoint_prefix", _checkpoint_prefix(args)])
    argv.extend(["--output_directory", args.output_dir])
    return argv

def _checkpoint_prefix(args):
    return _util.input_checkpoint(args.checkpoint, args.checkpoint_step)

def _export_tflite_ssd(config_path, args):
    from object_detection import export_tflite_ssd_graph
    sys.argv = _export_tflite_ssd_argv(config_path, args)
    log.info("Running export_tflite_ssd_graph with %s", sys.argv[1:])
    try:
        tf.app.run(export_tflite_ssd_graph.main)
    except SystemExit as e:
        if not e.code:
            _link_to_tflite_graph(args.output_dir)

def _export_tflite_ssd_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--trained_checkpoint_prefix", _checkpoint_prefix(args)])
    argv.extend(["--output_directory", args.output_dir])
    argv.extend(["--add_postprocessing_op", "true"])
    return argv

def _link_to_tflite_graph(output_dir):
    tflite_graph_name = "tflite_graph.pb"
    tflite_graph_path = os.path.join(output_dir, "tflite_graph.pb")
    link = os.path.join(output_dir, "frozen_inference_graph.pb")
    if not os.path.exists(tflite_graph_path):
        log.warning(
            "unable to create %s - %s does not exist",
            link, tflite_graph_path)
        return
    os.symlink(tflite_graph_name, link)

if __name__ == "__main__":
    main()
