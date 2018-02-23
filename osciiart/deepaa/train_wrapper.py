from __future__ import absolute_import
from __future__ import division

import argparse
import os

from guild.plugins import keras_op_main
from guild import util

import train as train_wrapped

def train():
    args = _parse_args()
    keras_op_main.patch_keras(args)
    if not os.path.exists("model"):
        os.makedirs("model")
    try:
        train_wrapped.train()
    except KeyboardInterrupt:
        print("Stopping workers...")
        util.kill_process_tree(os.getpid(), timeout=10)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_known_args()[0]

if __name__ == "__main__":
    train()
