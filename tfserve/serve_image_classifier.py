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
import io
import logging
import os
import sys

import numpy as np
import PIL
import tfserve

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

class Handler(object):

    def __init__(self, args):
        self.input_layer = args.input_layer
        self.image_width = args.image_width
        self.image_height = args.image_height

    def encode(self, data):
        try:
            return self._encode(data)
        except Exception as e:
            log.exception("handling data")

    def _encode(self, data):
        img_orig = PIL.Image.open(io.BytesIO(data))
        img_sized = img_orig.resize((self.image_width, self.image_height))
        img_input = np.asarray(img_sized) / 255
        return {
            self.input_layer: img_input
        }

    def decode(self, outputs):
        # TODO
        return {}

def main(argv):
    args = _init_args(argv)
    _serve(args)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graph", required=True,
        help="path to frozen model graph (required)")
    p.add_argument(
        "--labels", required=True,
        help="path to class labels (required)")
    p.add_argument(
        "--image-width", required=True, type=int,
        help="model input image width (required)")
    p.add_argument(
        "--image-height", required=True, type=int,
        help="model input image height (required)")
    p.add_argument(
        "--input-layer", required=True,
        help="model input layer (required)")
    p.add_argument(
        "--output-layer", required=True,
        help="model output layer (required)")
    p.add_argument(
        "--host", default=DEFAULT_HOST,
        help="host to bind server to (%s)" % DEFAULT_HOST)
    p.add_argument(
        "--port", default=DEFAULT_PORT, type=int,
        help="port to listen on (%i)" % DEFAULT_PORT)
    return p.parse_args(argv[1:])

def _serve(args):
    handler = Handler(args)
    app = tfserve.TFServeApp(
        args.graph,
        [args.input_layer], [args.output_layer],
        handler.encode,
        handler.decode)
    app.run(args.host, args.port)

if __name__ == "__main__":
    main(sys.argv)
