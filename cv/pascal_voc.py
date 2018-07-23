from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import random
import re
import sys

import click
from lxml import etree

import tensorflow as tf

try:
    from object_detection.utils import dataset_util
    from object_detection.utils import label_map_util
except ImportError as e:
    sys.stderr.write("WARNING: %s\n" % e)

def main():
    args = _parse_args()
    labels = _init_labels(args)
    train, val = _init_examples(args)
    _write_records("train.record", train, labels, args)
    _write_records("val.record", val, labels, args)

def _init_labels(args):
    return label_map_util.get_label_map_dict(args.labels)

def _init_examples(args):
    examples = _list_examples(args)
    random.seed(519)
    random.shuffle(examples)
    return _split_examples(examples, args)

def _list_examples(args):
    return [
        os.path.splitext(name)[0]
        for name in os.listdir(args.annotations_dir)
    ]

def _split_examples(examples, args):
    val = int(len(examples) * args.val_split)
    return examples[val:], examples[:val]

def _write_records(filename, examples, labels, args):
    path = os.path.join(args.output_dir, filename)
    writer = tf.python_io.TFRecordWriter(path)
    print("{} ({} examples):".format(path, len(examples)))
    if examples:
        with _progress(len(examples)) as bar:
            for example in examples:
                record = _init_record(example, labels, args)
                writer.write(record)
                bar.update(1)
    writer.close()

def _progress(length):
    bar = click.progressbar(length=length)
    bar.is_hidden = False
    return bar

def _init_record(example, labels, args):
    ann = _init_annotation(example, args)
    image_filename = ann["filename"]
    image_path = os.path.join(args.images_dir, image_filename)
    image_bytes = open(image_path, "rb").read()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    width = int(ann["size"]["width"])
    height = int(ann["size"]["height"])
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
        if args.object_class_method == "object":
            class_name = obj["name"]
        elif args.object_class_method == "filename":
            class_name = _object_class_from_filename(image_filename)
        else:
            raise AssertionError(args.object_class_method)
        class_text.append(class_name.encode())
        class_label.append(labels[class_name])
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
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def _init_annotation(example, args):
    path = os.path.join(args.annotations_dir, example + ".xml")
    node = etree.fromstring(open(path, "r").read())
    return dataset_util.recursive_parse_xml_to_dict(node)["annotation"]

def _object_class_from_filename(filename):
    m = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', filename, re.I)
    if not m:
        raise RuntimeError(
            "cannot get object class from filename: %s"
            % filename)
    return m.group(1)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        default="images",
        help="Directory containing images to prepare (images)")
    p.add_argument(
        "--annotations-dir",
        default="annotations",
        help="Directory containing image annotations (annotations)")
    p.add_argument(
        "--labels",
        default="labels.pbtxt",
        help="Path to label proto")
    p.add_argument(
        "--val-split",
        default=0.2,
        type=float,
        help="Percent of examples reserved for validation (0.2)")
    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write prepare dataset files (current directory)")
    p.add_argument(
        "--object-class-method",
        default="object",
        choices=["filename", "object"],
        help="Method of getting object class name")
    return p.parse_args()

if __name__ == "__main__":
    main()
