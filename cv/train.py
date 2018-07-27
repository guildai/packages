from __future__ import print_function

import argparse
import sys

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

try:
    from object_detection import model_main
except ImportError as e:
    sys.stderr.write("ERROR: %s\n" % e)

def main():
    args = _parse_args()
    sys.argv = _model_main_argv(args)
    tf.app.run(model_main.main)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir", metavar="DIR",
        help="directory to write model checkpoints and logs")
    p.add_argument(
        "config", metavar="CONFIG",
        help="path to detection config")
    return p.parse_args()

def _model_main_argv(args):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", args.config])
    if args.model_dir:
        argv.extend(["--model_dir", args.model_dir])
    return argv

if __name__ == "__main__":
    main()
