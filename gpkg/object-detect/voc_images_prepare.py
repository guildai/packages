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
import hashlib
import logging
import os
import random
import sys

from lxml import etree

import yaml

import tensorflow as tf

from object_detection.utils import dataset_util

from gpkg.slim import _tfrecord

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
        _examples(train, label_ids, args),
        len(train),
        args.output_dir,
        args.output_prefix,
        args.max_file_size,
        type_desc="train")
    _tfrecord.write_records(
        "val",
        _examples(val, label_ids, args),
        len(val),
        args.output_dir,
        args.output_prefix,
        args.max_file_size,
        type_desc="validation")
    _write_labels(label_ids, args)
    _write_dataset_config(len(label_ids), len(val), args)

def _init_args(argv):
    p = argparse.ArgumentParser(argv)
    p.add_argument(
        "--annotations-dir", metavar="DIR",
        required=True,
        help="directory containing image annotations (required)")
    p.add_argument(
        "--images-dir", metavar="DIR",
        required=True,
        help=(
            "directory containing images associated with annotations "
            "(required)"))
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
        "--config-data-path",
        default="data",
        help="path for data when generating dataset.yml (data)")
    p.add_argument(
        "--config-labels-path",
        default=".",
        help="path for labels when generating dataset.yml (.)")
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
    log.info("Reading examples from %s", args.annotations_dir)
    all_ann = _ordered_annotations(args.annotations_dir)
    if args.random_seed is not None:
        random.seed(args.random_seed)
    random.shuffle(all_ann)
    train_ann, val_ann = _split_ann(all_ann, args)
    if not train_ann or not val_ann:
        _error(
            "not enough examples to generate train "
            "and validation datasets - is val-split too "
            "low or too high?")
    train_labels, train = _load_examples(train_ann)
    val_labels, val = _load_examples(val_ann)
    label_ids = _label_map(train_labels.union(val_labels))
    return label_ids, train, val

def _ordered_annotations(dir):
    return sorted(glob.glob(os.path.join(dir, "*.xml")))

def _split_ann(ann, args):
    val = int(len(ann) * args.val_split / 100)
    return ann[val:], ann[:val]

def _load_examples(filenames):
    examples = []
    labels = set()
    for path in filenames:
        node = etree.fromstring(open(path, "r").read())
        data = dataset_util.recursive_parse_xml_to_dict(node)["annotation"]
        for obj in data.get("object") or ():
            labels.add(obj["name"])
        examples.append(data)
    return labels, examples

def _example_labels(examples):
    labels = set()
    for _image_path, obj in examples:
        labels.add(obj["name"])
    return labels

def _label_map(labels):
    return {name: (i + 1) for i, name in enumerate(sorted(labels))}

def _ensure_output_dir(args):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    else:
        log.debug("Created %s", args.output_dir)

def _examples(ann_data, label_ids, args):
    for ann in ann_data:
        yield None, _tf_example(ann, label_ids, args)

def _tf_example(ann, label_ids, args):
    image_filename = ann["filename"]
    image_path = os.path.join(args.images_dir, image_filename)
    image_bytes = open(image_path, "rb").read()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    width, height = _ann_size(ann)
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    class_text = []
    class_label = []
    difficult = []
    truncated = []
    poses = []
    for obj in ann["object"]:
        xmin.append(float(obj["bndbox"]["xmin"]) / width)
        xmax.append(float(obj["bndbox"]["xmax"]) / width)
        ymin.append(float(obj["bndbox"]["ymin"]) / height)
        ymax.append(float(obj["bndbox"]["ymax"]) / height)
        class_name = obj["name"]
        class_text.append(class_name.encode())
        class_label.append(label_ids[class_name])
        difficult.append(int(obj["difficult"]))
        truncated.append(int(obj["truncated"]))
        poses.append(obj["pose"].encode())
    feature = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(image_filename.encode()),
        "image/source_id": dataset_util.bytes_feature(image_filename.encode()),
        "image/key/sha256": dataset_util.bytes_feature(image_digest.encode()),
        "image/encoded": dataset_util.bytes_feature(image_bytes),
        "image/format": dataset_util.bytes_feature("jpeg".encode()),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmin),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmax),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymin),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymax),
        "image/object/class/text": dataset_util.bytes_list_feature(class_text),
        "image/object/class/label": dataset_util.int64_list_feature(class_label),
        "image/object/difficult": dataset_util.int64_list_feature(difficult),
        "image/object/truncated": dataset_util.int64_list_feature(truncated),
        "image/object/view": dataset_util.bytes_list_feature(poses),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _ann_size(ann):
    try:
        size = ann["size"]
    except KeyError:
        size = ann["size_part"]
    return int(size["width"]), int(size["height"])

def _write_labels(label_ids, args):
    labels_name = args.output_prefix + "labels.pbtxt"
    labels_path = os.path.join(args.output_dir, labels_name)
    log.info("Writing class labels %s", labels_path)
    id_to_name_map = {label_ids[name]: name for name in label_ids}
    with open(labels_path, "w") as f:
        _write_labels_proto(id_to_name_map, f)

def _write_labels_proto(names, f):
    for id in sorted(names):
        f.write(
            "item {\n"
            "  id: %s\n"
            "  name: %r\n"
            "}\n\n" % (id, names[id]))

def _write_dataset_config(labels, examples, args):
    config_path = os.path.join(
        args.output_dir,
        args.output_prefix + "dataset.yml")
    log.info("Writing dataset config %s", config_path)
    config = _dataset_config(labels, examples, args)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def _dataset_config(labels, examples, args):
    return {
        "num_classes": labels,
        "eval_config": {
            "metrics_set": [
                "coco_detection_metrics",
            ],
            "num_examples": examples
        },
        "train_input_reader": {
            "tf_record_input_reader": {
                "input_path": [
                    "%s/%strain-*-*.tfrecord"
                    % (args.config_data_path, args.output_prefix)
                ]
            },
            "label_map_path": (
                "%s/%slabels.pbtxt"
                % (args.config_labels_path, args.output_prefix))
        },
        "eval_input_reader": {
            "tf_record_input_reader": {
                "input_path": [
                    "%s/%sval-*-*.tfrecord"
                    % (args.config_data_path, args.output_prefix)
                ]
            },
            "label_map_path": (
                "%s/%slabels.pbtxt"
                % (args.config_labels_path, args.output_prefix)),
            "shuffle": False,
            # If this is not set to 1, object_detection may re-read
            # examples during eval.
            "num_readers": 1,
        }
    }

def _error(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
