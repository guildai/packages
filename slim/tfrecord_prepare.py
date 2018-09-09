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
import glob
import io
import logging
import os
import random
import sys
import time
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

    def __init__(self, output_dir, basename, example_count, max_file_size_mb):
        self.output_dir = output_dir
        self.basename = basename
        self.example_count = example_count
        self.max_file_size = (max_file_size_mb - 1) * 1024 * 1024
        self._writer = None
        self._writer_path = None
        self._cur_start = None
        self._cur_size = 0
        self._last_written = 0

    def write(self, example):
        example_bytes = example.SerializeToString()
        writer = self._next_writer(len(example_bytes))
        writer.write(example_bytes)
        self._last_written += 1
        self._cur_size += len(example_bytes)

    def _next_writer(self, next_len):
        if self._writer is None or self._next_too_big(next_len):
            self._new_writer()
        return self._writer

    def _next_too_big(self, next_len):
        return (
            self.max_file_size > 0 and
            self._cur_size + next_len > self.max_file_size)

    def _new_writer(self):
        self.close()
        self._cur_start = self._last_written + 1
        path = os.path.join(self.output_dir, self._tfrecord_name())
        self._writer = tf.python_io.TFRecordWriter(path)
        self._writer_path = path

    def _tfrecord_name(self):
        digits_needed = self._digits_needed(self.example_count)
        digits_pattern = "%%0.%ii" % digits_needed
        start = digits_pattern % self._cur_start
        if self._last_written > 0:
            end = digits_pattern % self._last_written
        else:
            end = "?" * digits_needed
        return "%s-%s-%s.tfrecord" % (self.basename, start, end)

    @staticmethod
    def _digits_needed(n):
        digits = 1
        while n > 10:
            digits += 1
            n = n // 10
        return digits

    def close(self, rename=True):
        if self._writer is not None:
            self._writer.close()
            if rename:
                self._rename_writer()
            self._writer = None
            self._writer_path = None
            self._cur_size = 0
            self._writer_path = None

    def _rename_writer(self):
        assert self._writer is not None
        new_path = os.path.join(self.output_dir, self._tfrecord_name())
        assert new_path != self._writer_path, self._writer_path
        os.rename(self._writer_path, new_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _value, _tb):
        self.close(exc_type is None)

def main(argv):
    args = _parse_args(argv)
    _init_logging(args)
    _check_existing_output(args)
    label_ids, train, val = _init_examples(args)
    log.info(
        "Found %i examples of %i classes",
        len(train) + len(val), len(label_ids))
    _ensure_output_dir(args)
    _write_labels(label_ids, args)
    _write_records("train", "train", train, label_ids, args)
    _write_records("validation", "val", val, label_ids, args)

def _init_logging(args):
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format="%(message)s", level=level)

def _check_existing_output(args):
    g = lambda pattern: os.path.join(
        args.output_dir, pattern % args.output_prefix)
    globs = (
        g("%strain-*.tfrecord"),
        g("%sval-*.tfrecord"),
        g("%slabels.txt"),
    )
    matches = []
    for pattern in globs:
        matches.extend(glob.glob(pattern))
    if matches:
        _error(
            "the following record files already exist in %s: %s"
            % (args.output_dir, ", ".join(matches)))

def _init_examples(args):
    log.info("Reading examples from %s", args.images_dir)
    label_ids, filenames = _list_images(args.images_dir)
    random.seed(args.random_seed)
    random.shuffle(filenames)
    train, val = _split_examples(filenames, args)
    if not train or not val:
        _error(
            "not enough examples to generate train "
            "and validation datasets")
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
        acc.append((label, path))

def _label_map(labels):
    return {name: label_id for label_id, name in enumerate(sorted(labels))}

def _split_examples(examples, args):
    val = int(len(examples) * args.val_split / 100)
    return examples[val:], examples[:val]

def _ensure_output_dir(args):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    else:
        log.debug("Created %s", args.output_dir)

def _write_labels(label_ids, args):
    labels_name = args.output_prefix + "labels.txt"
    log.info(
        "Writing class labels %s",
        os.path.join(args.output_dir, labels_name))
    id_to_name_map = {label_ids[name]: name for name in label_ids}
    dataset_utils.write_label_file(id_to_name_map, args.output_dir, labels_name)

def _write_records(type_desc, basename, examples, labels, args):
    writer = Writer(
        args.output_dir,
        args.output_prefix + basename,
        len(examples),
        args.max_file_size)
    with writer:
        log.info(
            "Writing %i %s records %s",
            len(examples), type_desc, _filename_pattern(basename, args))
        quiet = os.getenv("NO_PROGRESS") == "1"
        with _progress(len(examples)) as bar:
            _progress_start(bar)
            for label, path in examples:
                example = _image_tf_example(path, labels[label])
                writer.write(example)
                if not quiet:
                    bar.update(1)
            _progress_finish(bar)

def _filename_pattern(basename, args):
    return os.path.join(
        args.output_dir,
        "%s%s-*.tfrecord" % (args.output_prefix, basename))

def _progress(length):
    bar = click.progressbar(length=length)
    bar.is_hidden = False
    return bar

def _progress_start(_bar):
    # Workaround progress not shown on small datasets - click progress
    # bar apparently needs some time to setup.
    time.sleep(0.5)

def _progress_finish(_bar):
    # Workaround stdout sync with other messages - click progress bar
    # needs some time to stop writing.
    time.sleep(0.1)

def _image_tf_example(image_path, label_id):
    image_bytes, image_format, image_h, image_w = _load_image(image_path)
    log.debug(
        "%s: format=%s size=%i height=%i width=%i",
        image_path, image_format, len(image_bytes), image_h, image_w)
    return dataset_utils.image_to_tfexample(
        image_bytes,
        image_format.encode(),
        image_h,
        image_w,
        label_id)

def _load_image(path):
    image_bytes = open(path, "rb").read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image_bytes, image.format, image.height, image.width

def _error(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)

def _parse_args(argv):
    p = argparse.ArgumentParser(argv)
    p.add_argument(
        "images_dir", metavar="IMAGES-DIR",
        help="directory containing images to prepare")
    p.add_argument(
        "-s", "--val-split", metavar="N",
        default=30,
        type=int,
        help="percent of examples reserved for validation (default is 30)")
    p.add_argument(
        "-p", "--output-prefix", metavar="PREFIX",
        default="",
        help="optional prefix to use for generated database files")
    p.add_argument(
        "-o", "--output-dir", metavar="DIR",
        default=".",
        help=(
            "directory to write generated dataset files info "
            "(default is current directory)"))
    p.add_argument(
        "-r", "--random-seed", metavar="N",
        default=829, # arbitrary constant
        type=int,
        help="seed used to randomly split training and validation images")
    p.add_argument(
        "-m", "--max-file-size", metavar="MB",
        default=100,
        type=int,
        help=(
            "max size per TF record file in MB; use 0 to disable "
            "(default is 100)"))
    p.add_argument(
        "--debug", action="store_true",
        help="show debug info")
    return p.parse_args()

if __name__ == "__main__":
    main(sys.argv)
