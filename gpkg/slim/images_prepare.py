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

import PIL

from slim.datasets import dataset_utils

import _tfrecord

log = logging.getLogger()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

def main(argv):
    args = _init_args(argv)
    _init_logging(args)
    _check_existing_output(args)
    label_ids, train, val = _init_examples(args)
    log.info(
        "Found %i examples of %i classes",
        len(train) + len(val), len(label_ids))
    _ensure_output_dir(args)
    _tfrecord.write_records(
        "train",
        _examples(train, label_ids),
        len(train),
        args.output_dir, args.output_prefix,
        args.max_file_size, True, "train")
    _tfrecord.write_records(
        "val",
        _examples(val, label_ids),
        len(val),
        args.output_dir, args.output_prefix,
        args.max_file_size, False, "validation")
    _write_labels(label_ids, args)

def _init_args(argv):
    p = argparse.ArgumentParser(argv)
    p.add_argument(
        "-i", "--images-dir", metavar="DIR",
        required=True,
        help="directory containing images to prepare (required)")
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
    label_ids, filenames = _ordered_images(args.images_dir)
    if args.random_seed is not None:
        random.seed(args.random_seed)
    random.shuffle(filenames)
    train, val = _split_examples(filenames, args)
    if not train or not val:
        _error(
            "not enough examples to generate train "
            "and validation datasets")
    return label_ids, train, val

def _ordered_images(root):
    labels = set()
    filenames = []
    for name in sorted(os.listdir(root)):
        label_dir = os.path.join(root, name)
        if os.path.isdir(label_dir):
            labels.add(name)
            _apply_ordered_filenames(label_dir, name, filenames)
    return _label_map(labels), filenames

def _apply_ordered_filenames(root, label, acc):
    for name in sorted(os.listdir(root)):
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
    return {name: i for i, name in enumerate(sorted(labels))}

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

def _examples(label_paths, label_ids):
    for label, path in label_paths:
        yield label, _tf_example(path, label_ids[label])

def _tf_example(image_path, label_id):
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

def _write_labels(label_ids, args):
    labels_name = args.output_prefix + "labels.txt"
    log.info(
        "Writing class labels %s",
        os.path.join(args.output_dir, labels_name))
    id_to_name_map = {label_ids[name]: name for name in label_ids}
    dataset_utils.write_label_file(id_to_name_map, args.output_dir, labels_name)

def _error(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
