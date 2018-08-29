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
import os
import random

import click

import tensorflow as tf

class Writer(object):

    def __init__(self, output_dir, basename, images_per_shard=None):
        self.output_dir = output_dir
        self.basename = basename
        self.images_per_shard = images_per_shard
        self._shard_num = None
        self._writer = None
        self._shard_images = 0

    def write(self, example):
        writer = self._next_writer()
        writer.write(example.SerializeToString())

    def _next_writer(self):
        if self._writer is None:
            assert self.images_per_shard is None, self.images_per_shard
            path = os.path.join(self.output_dir, self.basename)
            self._writer = tf.python_io.TFRecordWriter(path)
        return self._writer

    def close(self):
        if self._writer is not None:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _tb):
        self.close()

def main():
    args = _parse_args()
    labels, train, val = _init_examples(args)
    _write_records("train.record", train, labels, args)
    _write_records("val.record", val, labels, args)

def _init_examples(args):
    filenames = _list_images(args)
    random.seed(args.random_seed)
    random.shuffle(filenames)
    labels = _init_labels(filenames)
    train, val = _split_examples(filenames, args)
    return labels, train, val

def _init_labels(_filenames):
    # TODO: label map from filenames
    return {}

def _list_images(args):
    for root, dirs, files in os.walk(args.images_dir):
        print(root, dirs, files)
    return []

def _split_examples(examples, args):
    val = int(len(examples) * args.val_split)
    return examples[val:], examples[:val]

def _write_records(basename, examples, labels, args):
    writer = Writer(basename, args)
    with Writer(basename, args) as writer:
        with _progress(len(examples)) as bar:
            for path in examples:
                full_path = os.path.join(args.images_dir, path)
                writer.write(_tf_example(full_path, labels))
                bar.update(1)

def _progress(length):
    bar = click.progressbar(length=length)
    bar.is_hidden = False
    return bar

def _tf_example(_path, _labels):
    # TODO: load image from path and configure example
    return None

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        default="images",
        help="Directory containing images to prepare (images)")
    p.add_argument(
        "--val-split",
        default=0.3,
        type=float,
        help="Percent of examples reserved for validation (0.3)")
    p.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix to use for generated database files.")
    p.add_argument(
        "--output-dir",
        default=".",
        help=(
            "Directory to write generated dataset files info "
            "(current directory)"))
    p.add_argument(
        "--random-seed",
        default=829, # arbitrary constant
        type=int,
        help="Seed used to randomly split training and validation images")
    p.add_argument(
        "--images-per-shard",
        type=int,
        help="Max images per TF record shard (default is to not use shards)")
    return p.parse_args()

if __name__ == "__main__":
    main()
