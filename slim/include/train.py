from __future__ import absolute_import
from __future__ import division

import os
import sys

import tensorflow as tf

from tensorflow.contrib import slim

sys.path.insert(0, "slim")

from datasets import dataset_factory
from datasets import dataset_utils

import train_image_classifier

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
    dataset_factory.datasets_map["custom"] = CustomDataset()
    train_image_classifier.main([])

if __name__ == "__main__":
    main()
