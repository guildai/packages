from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from object_detection import model_main

def main():
    args = _init_args()
    sys.argv = _model_main_argv(args)
    tf.app.run(model_main.main)

def _init_args():
    p = argparse.ArgumentParser()
    return p.parse_args()

def _model_main_argv(args):
    print(args)
    return [
        sys.argv[0],
        # TODO: either generate config or get from args
        "--pipeline_config_path", "pet-detector.config"]

if __name__ == "__main__":
    main()
