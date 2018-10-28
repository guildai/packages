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

def patch_all():
    patch_4780()

def patch_4780():
    # TypeError: can't pickle dict_values objects
    # https://github.com/tensorflow/models/issues/4780
    from object_detection import eval_util
    f0 = eval_util.get_eval_metric_ops_for_evaluators
    def f(evaluation_metrics, categories, eval_dict):
        return f0(evaluation_metrics, list(categories), eval_dict)
    eval_util.get_eval_metric_ops_for_evaluators = f
