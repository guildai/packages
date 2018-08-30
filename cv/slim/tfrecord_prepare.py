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
import io
import logging
import os
import random
import warnings

import click
import PIL

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    import tensorflow as tf

from slim.datasets import dataset_utils

log = logging.getLogger()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

class Writer(object):

    def __init__(self, output_dir, basename, max_file_size_mb):
        self.output_dir = output_dir
        self.basename = basename
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._writer = None
        self._writer_path = None
        self._cur_start = None
        self._cur_size = 0
        self._last_written = 0

    def write(self, example):
        example_bytes = example.SerializeToString()
        writer = self._next_writer(len(example_bytes))
        writer.write()
        self._last_written += 1

    def _next_writer(self, next_write_size):
        if self._writer is None:
            self._new_writer()
        elif self._cur_size + next_write_size >= self.max_file_size:
            self._new_writer()
        return self._writer

    def _new_writer(self):
        self.close()
        self._cur_start = self._last_written + 1
        path = os.path.join(self.output_dir, self._tfrecord_name())
        self._writer = tf.python_io.TFRecordWriter(path)
        self._writer_path = path

    def _tfrecord_name(self):
        start = "%0.6i" % self._cur_start
        if self._last_written > 0:
            end = "%0.6i" % self._last_written
        else:
            end = "?" * 6
        return "%s-%s-%s.tfrecord" % (self.basename, start, end)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._rename_writer()
            self._writer = None
            self._writer_path = None
            self._cur_size = 0
            self._writer_path = None

    def _rename_writer(self):
        assert self._writer is not None
        new_path = os.path.join(self.output_dir, self._tfrecord_name())
        assert new_path != self._writer_path, self._writer_path
        os.move(self._writer_path, new_path)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _tb):
        self.close()

def main():
    args = _parse_args()
    _init_logging(args)
    label_ids, train, val = _init_examples(args)
    _write_records("train", train, label_ids, args)
    _write_records("val", val, label_ids, args)

def _init_logging(args):
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format="%(message)s", level=level)

def _init_examples(args):
    log.info("Reading images from %s", args.images_dir)
    label_ids, filenames = _list_images(args.images_dir)
    random.seed(args.random_seed)
    random.shuffle(filenames)
    train, val = _split_examples(filenames, args)
    return label_ids, train, val

def _list_images(root):
    labels = set()
    filenames = []
    for name in os.listdir(root):
        label_dir = os.path.join(root, name)
        if os.path.isdir(label_dir):
            labels.add(name)
            _apply_filenames(label_dir, name, filenames)
    return _label_map(labels), filenames

def _apply_filenames(root, label, acc):
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            log.warning("ignoring directory %s in %s", name, root)
            continue
        _, ext = os.path.splitext(name)
        if ext not in IMAGE_EXTENSIONS:
            log.warning(
                "ignoring file %s in %s (unsupported extension)",
                name, root)
            continue
        log.debug("Adding %s", path)
        acc.append((label, _format_for_ext(ext), path))

def _format_for_ext(ext):
    if ext == ".jpeg":
        return "jpg"
    else:
        return ext[1:]

def _label_map(labels):
    return {name: label_id for label_id, name in enumerate(sorted(labels))}

def _split_examples(examples, args):
    val = int(len(examples) * args.val_split / 100)
    return examples[val:], examples[:val]

def _write_records(basename, examples, labels, args):
    writer = Writer(
        args.output_dir,
        args.output_prefix + basename,
        args.max_file_size)
    with writer:
        with _progress(len(examples)) as bar:
            for label, fmt, path in examples:
                import pdb;pdb.set_trace()
                full_path = os.path.join(args.images_dir, path)
                example = _image_tf_example(full_path, fmt, labels[label])
                writer.write(example)
                bar.update(1)

def _progress(length):
    bar = click.progressbar(length=length)
    bar.is_hidden = False
    return bar

def _image_tf_example(image_path, image_format, label_id):
    image_bytes, image_h, image_w = _load_image(image_path)
    return dataset_utils.image_to_tfexample(
        image_bytes,
        image_format,
        image_h,
        image_w,
        label_id)

def _load_image(path):
    image_bytes = open(path, "r").read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image_bytes, image.height, image.width

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "images_dir", metavar="IMAGES-DIR",
        help="directory containing images to prepare")
    p.add_argument(
        "--val-split", metavar="N",
        default=30,
        type=int,
        help="percent of examples reserved for validation (30)")
    p.add_argument(
        "--output-prefix", metavar="VAL",
        default="",
        help="optional prefix to use for generated database files")
    p.add_argument(
        "--output-dir", metavar="DIR",
        default=".",
        help=(
            "directory to write generated dataset files info "
            "(current directory)"))
    p.add_argument(
        "--random-seed", metavar="N",
        default=829, # arbitrary constant
        type=int,
        help="seed used to randomly split training and validation images")
    p.add_argument(
        "-m", "--max-file-size", metavar="MB",
        default=100,
        type=int,
        help="max size per TF record file in MB (100)")
    p.add_argument(
        "--debug", action="store_true",
        help="show debug info")
    return p.parse_args()

if __name__ == "__main__":
    main()
