from __future__ import absolute_import
from __future__ import division

import argparse
import imp
import os
import re
import sys

import tensorflow as tf

from tensorflow.contrib import slim

sys.path.insert(0, "slim")

from datasets import dataset_factory
from datasets import dataset_utils

class CustomDataset(object):

    def get_split(self, split_name, dataset_dir, _file_pattern, reader):
        file_pattern = os.path.join(dataset_dir, "%s_*.tfrecord" % split_name)
        reader = reader or tf.TFRecordReader
        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/format": tf.FixedLenFeature((), tf.string, default_value="png"),
            "image/class/label": tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        items_to_handlers = {
            "image": slim.tfexample_decoder.Image(),
            "label": slim.tfexample_decoder.Tensor("image/class/label"),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
        split_counts = self._read_split_counts(dataset_dir)
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_counts[split_name],
            items_to_descriptions={},
            num_classes=len(labels_to_names),
            labels_to_names=labels_to_names)

    @staticmethod
    def _read_split_counts(dataset_dir):
        counts = {}
        path = os.path.join(dataset_dir, "splits.txt")
        for line in open(path, "r").readlines():
            name, count_str = line.split(":")
            counts[name] = int(count_str)
        return counts

def main():
    main_mod = sys.argv[1]
    sys.argv = _init_argv(main_mod, sys.argv[:1] + sys.argv[2:])
    dataset_factory.datasets_map["custom"] = CustomDataset()
    mod_info = imp.find_module(main_mod)
    # need stack variable to function as globals are reset on
    # load_module
    handle_interrupted = _handle_interrupted
    try:
        imp.load_module("__main__", *mod_info)
    except KeyboardInterrupt:
        handle_interrupted(main_mod)

def _init_argv(main_mod, args):
    if main_mod == "tensorflow/python/tools/freeze_graph":
        return _resolve_input_checkpoint_arg(args)
    else:
        return args

def _resolve_input_checkpoint_arg(args):
    p = argparse.ArgumentParser()
    p.add_argument("--input_checkpoint")
    parsed, rest = p.parse_known_args(args)
    if parsed.input_checkpoint:
        resolved = _resolve_checkpoint_path(parsed.input_checkpoint)
        return rest + ["--input_checkpoint", resolved]
    else:
        return args

def _resolve_checkpoint_path(path):
    if os.path.exists(path):
        return path
    return _model_checkpoint_path(os.path.dirname(path))

def _model_checkpoint_path(dir):
    index = os.path.join(dir, "checkpoint")
    if not os.path.exists(index):
        sys.stderr.write(
            "ERROR: cannot resolve model checkpoint: {} does not "
            "exist\n".format(index))
        sys.exit(1)
    line1 = open(index, "r").readlines()[0]
    path_val = re.search("\"(.+?)\"", line1).group(1)
    return os.path.join(dir, path_val)

def _handle_interrupted(main_mod):
    # We need to re-import here, assuming this is called after
    # imp.load_module, which resets our globals
    import os, sys
    sys.stderr.write("Run terminated")
    if main_mod == "train_image_classifier":
        sys.stderr.write(
            " - refer to %s for model checkpoints, which may be "
            "used to resume training" % os.path.abspath("."))
    sys.stderr.write("\n")
    sys.exit(1)

if __name__ == "__main__":
    main()
