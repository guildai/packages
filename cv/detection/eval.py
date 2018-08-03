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

import config_util

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def main():
    args = _parse_args()
    _validate_args(args)
    config_path = _init_config(args)
    if os.getenv("SKIP_EVAL") == "1":
        log.info("SKIP_EVAL set, skipping evaluate")
        return
    sys.argv = _model_main_argv(config_path, args)
    log.info("Running model_main with %s", sys.argv[1:])
    tf.app.run(model_main.main)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir", metavar="PATH", default="model",
        help="directory to write logs")
    p.add_argument(
        "--checkpoint-dir", metavar="PATH", default="checkpoint",
        help="directory containing checkpoint to evaluate")
    p.add_argument(
        "--eval-examples", metavar="COUNT", type=int,
        help="eval examples")
    config_util.add_config_args(p)
    return p.parse_args()

def _validate_args(args):
    config_util.validate_config_args(args)

def _init_config(args):
    try:
        return config_util.init_config(args)
    except config_util.ConfigError as e:
        sys.stderr.write("%s\n" % e)
        sys.exit(1)

def _model_main_argv(config_path, args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    argv.extend(["--model_dir", args.model_dir])
    argv.extend(["--checkpoint_dir", args.checkpoint_dir])
    argv.append("--run_once")
    return argv

if __name__ == "__main__":
    main()
