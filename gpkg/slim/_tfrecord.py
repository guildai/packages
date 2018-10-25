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

import collections
import logging
import os
import time
import warnings

import click

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    import tensorflow as tf

log = logging.getLogger()

class Writer(object):

    def __init__(self, output_dir, basename, examples_count, max_file_size_mb):
        self.output_dir = output_dir
        self.basename = basename
        self.examples_count = examples_count
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
        digits_needed = self._digits_needed(self.examples_count)
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

def write_records(basename,
                  examples,
                  examples_count,
                  output_dir,
                  output_prefix,
                  max_file_size=100,
                  write_weights=False,
                  type_desc=None):
    type_desc = type_desc or basename
    writer = Writer(
        output_dir,
        output_prefix + basename,
        examples_count,
        max_file_size)
    label_counts = collections.Counter()
    with writer:
        pattern = _filename_pattern(basename, output_dir, output_prefix)
        log.info(
            "Writing %i %s records %s",
            examples_count, type_desc, pattern)
        quiet = os.getenv("NO_PROGRESS") == "1"
        with _progress(examples_count) as bar:
            _progress_start(bar)
            for label, example in examples:
                writer.write(example)
                if write_weights:
                    label_counts.update([label])
                if not quiet:
                    bar.update(1)
            _progress_finish(bar)
    if write_weights:
        _write_weights(basename, label_counts, output_dir, output_prefix)

def _write_weights(basename, label_counts, output_dir, output_prefix):
    weights = _balanced_label_weights(label_counts)
    weights_file = os.path.join(
        output_dir,
        output_prefix + basename + "-weights.txt")
    log.info("Writing class weights %s", weights_file)
    with open(weights_file, "w") as f:
        for name in sorted(weights):
            f.write("%s:%f\n" % (name, weights[name]))

def _balanced_label_weights(counts):
    class_count = len(counts)
    total_count = sum(counts.values())
    return {
        name: total_count / (class_count * counts[name])
        for name in counts
    }

def _filename_pattern(basename, output_dir, output_prefix):
    return os.path.join(
        output_dir,
        "%s%s-*.tfrecord" % (output_prefix, basename))

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
