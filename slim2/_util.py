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

import os
import sys

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
