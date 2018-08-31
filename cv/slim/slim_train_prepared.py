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

import sys

import tensorflow as tf

sys.path.insert(0, "slim")

import train_image_classifier

from datasets import dataset_factory

def main(argv):
    if "--dataset_name" in argv:
        _error("--dataset_name is not supported")
    if "--model_name" not in argv:
        _error("--model_name is required")
    dataset_factory.datasets_map = {
        "custom": globals()
    }
    argv = argv + ["--dataset_name", "custom"]
    tf.app.run(train_image_classifier.main, argv)

def get_split(split_name, dataset_dir):
    reader = reader or tf.TFRecordReader

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)


    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    tfrecord_pattern = os.path.join(
        dataset_dir,
        "*{}-*-*.tfrecord" % split_name)

    return slim.dataset.Dataset(
        data_sources=tfrecord_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)

def _error(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
