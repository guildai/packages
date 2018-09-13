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

import glob
import os
import re

import tensorflow as tf

from datasets import dataset_factory
from datasets import dataset_utils

import _util

def patch_dataset_factory():
    dataset_factory.datasets_map = {
        "custom": __import__(__name__)
    }

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    assert file_pattern is None, file_pattern
    assert reader is None, reader
    source_pattern = _source_pattern(split_name, dataset_dir)
    example_count = _example_count_from_sources(source_pattern)
    labels = dataset_utils.read_label_file(dataset_dir)
    return tf.contrib.slim.dataset.Dataset(
        data_sources=source_pattern,
        reader=tf.TFRecordReader,
        decoder=_decoder(),
        num_samples=example_count,
        items_to_descriptions=_item_descriptions(),
        num_classes=len(labels),
        labels_to_names=labels)

def _source_pattern(split_name, dataset_dir):
    pattern_map = {
        "validation": "*val-*-*.tfrecord",
        "train": "*train-*-*.tfrecord"
    }
    try:
        pattern = pattern_map[split_name]
    except KeyError:
        _util.error("unsupported split name: %r" % split_name)
    return os.path.join(dataset_dir, pattern)

def _example_count_from_sources(source_pattern):
    sources = glob.glob(source_pattern)
    if not sources:
        _util.error("no files matching '%s'" % source_pattern)
    count = 0
    for source in sources:
        m = re.search("[0-9]+-([0-9]+)", source)
        if m:
            count = max(count, int(m.group(1)))
    if count == 0:
        _util.error(
            "could not get example count from files '%s'"
            % source_pattern)
    return count

def _decoder():
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),
    }
    items_to_handlers = {
        "image": tf.contrib.slim.tfexample_decoder.Image(),
        "label": tf.contrib.slim.tfexample_decoder.Tensor("image/class/label"),
    }
    return tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features,
        items_to_handlers)

def _item_descriptions():
    return {
        "image": "Image",
        "label": "Image label",
    }
