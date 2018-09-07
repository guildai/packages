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
import sys

import tensorflow as tf

def error(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)

class argv(object):

    def __init__(self, new_argv):
        self._new_argv = new_argv
        self._argv_save = None

    def __enter__(self):
        self._argv_save = sys.argv
        sys.argv = self._new_argv
        if os.getenv("DEBUG") == "1":
            sys.stderr.write("DEBUG: sys.argv changed: %r\n" % sys.argv)

    def __exit__(self, exc_type, exc, tb):
        assert self._argv_save is not None, self._argv_save
        sys.argv = self._argv_save
        if os.getenv("DEBUG") == "1":
            sys.stderr.write("DEBUG: sys.argv restored: %r\n" % sys.argv)
        return False

def input_checkpoint(path, step=None):
    if step is None:
        return _latest_input_checkpoint(path)
    else:
        return _step_input_checkpoint(path, step)

def _latest_input_checkpoint(path):
    if os.path.isdir(path):
        checkpoint = tf.train.latest_checkpoint(path)
    elif path.endswith(".ckpt"):
        checkpoint = _latest_checkpoint_for_base(path)
    else:
        checkpoint = path
    if not checkpoint:
        error("cannot find latest checkpoint for path: %s" % path)
    return checkpoint

def _latest_checkpoint_for_base(base):
    max = 0
    latest = None
    for match in glob.glob("%s-*.meta" % base):
        match_step = _checkpoint_step(match)
        if match_step > max:
            max = match_step
            latest = match
    return os.path.splitext(latest)[0]

def _checkpoint_step(meta_path):
    m = re.search(r"-([0-9]+)\.meta$", meta_path)
    assert m, meta_path
    return int(m.group(1))

def _step_input_checkpoint(path, step):
    if os.path.isdir(path):
        latest = tf.train.latest_checkpoint(path)
        if not latest:
            error("cannot find a checkpoint in %s" % path)
        return _replace_step(latest, step)
    elif path.endswith(".ckpt"):
        return "%s-%i" % (path, step)
    else:
        error(
            "input checkpoint must be a directory or "
            "end in '.ckpt' when step is specified")

def _replace_step(path, step):
    assert re.search(r"-[0-9]+$", path), path
    return re.sub(r"-[0-9]+$", "-%i" % step, path)
