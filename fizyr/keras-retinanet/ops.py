from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import logging
import os
import shutil
import sys
import tarfile

log = logging.getLogger("guild")

from keras_retinanet.bin import train as trainlib

from guild.plugins import keras_op_main

def main(argv):
    keras_op_main.patch_keras()
    base_args, cmd_args = init_args(argv)
    if base_args.cmd == "train":
        train(base_args, cmd_args)
    elif base_args.cmd == "pascal-prepare":
        prepare_pascal()
    else:
        raise AssertionError(base_args.cmd)

def init_args(argv):
    p = argparse.ArgumentParser(argv)
    p.add_argument("cmd", choices=[
        "train",
        "pascal-prepare",
    ])
    p.add_argument("--dataset")
    p.add_argument("--dataset-path")
    p.add_argument("--weights")
    return p.parse_known_args()

def train(base_args, cmd_args):
    args = (
        cmd_args +
        _weights_opts(base_args) +
        [base_args.dataset, base_args.dataset_path]
    )
    log.info("Running train with args: %s", args)
    trainlib.main(args)

def _weights_opts(args):
    weights = args.weights
    if not weights:
        return []
    elif os.path.exists(weights):
        return ["--weights", weights]
    elif weights == "imagenet":
        return ["--imagenet-weights"]
    elif weights == "none":
        return ["--no-weights"]
    elif weights == "resume":
        # TODO: find latest train run and use snapshot
        return []
    else:
        # TODO: assume is run ID - use to find snapshot
        return []

def prepare_pascal():
    for path in glob.glob("*.tar"):
        print("Extracting %s" % path)
        f = tarfile.TarFile(path)
        f.extractall()

if __name__ == "__main__":
    main(sys.argv)
