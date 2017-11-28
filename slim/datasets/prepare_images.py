from __future__ import absolute_import
from __future__ import division

import argparse
import math
import os
import random
import sys

import tensorflow as tf

from slim.datasets import dataset_utils

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")

class State(object):

    def __init__(self, args):
        self.images_dir = args.images_dir
        self.output_dir = args.output_dir
        self.validation_split = args.validation_split
        self.shard_count = args.shards
        self._init_filenames_and_classes()
        self._init_filename_split()

    def _init_filenames_and_classes(self):
        dirs = []
        classes = []
        for name in os.listdir(self.images_dir):
            path = os.path.join(self.images_dir, name)
            if os.path.isdir(path):
                dirs.append(path)
                classes.append(name)
        filenames = []
        for dir in dirs:
            for name in os.listdir(dir):
                _, ext = os.path.splitext(name)
                if ext.lower() in IMAGE_EXTENSIONS:
                    filenames.append(os.path.join(dir, name))
        self.filenames = filenames
        self.classes = sorted(classes)
        self.class_id_map = dict(zip(classes, range(len(classes))))

    def _init_filename_split(self):
        random.seed(0)
        random.shuffle(self.filenames)
        validation_count = int(len(self.filenames) * self.validation_split)
        self.training = self.filenames[validation_count:]
        self.validation = self.filenames[:validation_count]

class ShardInfo(object):

    def __init__(self, filenames, shard_count):
        self.shard_count = shard_count
        self.images_per_shard = int(math.ceil(len(filenames) / shard_count))

class ImageReader(object):

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(
            self._decode_jpeg,
            feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def main():
    args = _parse_args()
    state = State(args)
    os.makedirs(state.output_dir)
    _convert_images(state.training, "train", state)
    _convert_images(state.validation, "validation", state)
    _write_label_file(state)
    _write_splits(state)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--validation-split", type=float, default=0.1)
    p.add_argument("--shards", type=int, default=5)
    return p.parse_args()

def _convert_images(filenames, split_name, state):
    shard_info = ShardInfo(filenames, state.shard_count)
    with tf.Graph().as_default():
        reader = ImageReader()
        with tf.Session("") as sess:
            for shard_id in range(shard_info.shard_count):
                _create_tfrecord(
                    split_name, shard_id, filenames,
                    shard_info, reader, sess, state)

def _create_tfrecord(split_name, shard_id, filenames, shard_info,
                     reader, sess, state):
    out_filename = _record_filename(split_name, shard_id, shard_info, state)
    with tf.python_io.TFRecordWriter(out_filename) as writer:
        for i in _shard_files_range(shard_id, shard_info, filenames):
            _status(
                "\r>> Converting image %d/%d shard %d"
                % (i + 1, len(filenames), shard_id))
            _write_shard_file(filenames[i], reader, writer, sess, state)
        _status()

def _shard_files_range(shard_id, shard_info, filenames):
    start = shard_id * shard_info.images_per_shard
    end = min((shard_id + 1) * shard_info.images_per_shard, len(filenames))
    return range(start, end)

def _write_shard_file(filename, reader, writer, sess, state):
    image_data = tf.gfile.FastGFile(filename, "rb").read()
    height, width = reader.read_image_dims(sess, image_data)
    class_name = os.path.basename(os.path.dirname(filename))
    class_id = state.class_id_map[class_name]
    _, ext = os.path.splitext(filename)
    example = dataset_utils.image_to_tfexample(
        image_data, ext[1:], height, width, class_id)
    writer.write(example.SerializeToString())

def _record_filename(split_name, shard_id, shard_info, state):
    name = (
        "%s_%05d-of-%05d.tfrecord"
        % (split_name, shard_id, shard_info.shard_count))
    return os.path.join(state.output_dir, name)

def _write_label_file(state):
    class_label_map = dict((id, cls) for cls, id in state.class_id_map.items())
    dataset_utils.write_label_file(class_label_map, state.output_dir)

def _write_splits(state):
    path = os.path.join(state.output_dir, "splits.txt")
    with open(path, "w") as f:
        f.write("train:%i\n" % len(state.training))
        f.write("validation:%i\n" % len(state.validation))

def _status(msg="\n"):
    sys.stdout.write(msg)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
