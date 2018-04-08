from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import shutil
import tarfile

from keras_retinanet.bin import train as trainlib

def main():
    base_args, cmd_args = init_args()
    if base_args.cmd == "train":
        train(base_args, cmd_args)
    elif base_args.cmd == "pascal-prepare":
        prepare_pascal()
    else:
        raise AssertionError(base_args.cmd)

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=[
        "train",
        "pascal-prepare",
    ])
    p.add_argument("--dataset")
    p.add_argument("--dataset-path")
    return p.parse_known_args()

def train(base_args, cmd_args):
    args = cmd_args + [base_args.dataset, base_args.dataset_path]
    trainlib.main(args)

def prepare_pascal():
    for path in glob.glob("*.tar"):
        print("Extracting %s" % path)
        f = tarfile.TarFile(path)
        f.extractall()

if __name__ == "__main__":
    main()
