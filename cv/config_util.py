from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import yaml

from google.protobuf import text_format

from object_detection.protos import pipeline_pb2

from guild import op_util

log = logging.getLogger()

CONFIG_FILENAME = "generated.config"

class ConfigError(Exception):
    pass

def add_config_args(p):
    p.add_argument(
        "--pipeline-config", metavar="PATH",
        help=("path to pbtxt formatted detection config - "
              "overrides all other config"))
    p.add_argument(
        "--model-config", metavar="PATH",
        help="path to YAML formatted model config")
    p.add_argument(
        "--train-config", metavar="PATH",
        help="path to YAML formatted train config")
    p.add_argument(
        "--eval-config", metavar="PATH",
        help="path to YAML formatted eval config")
    p.add_argument(
        "--dataset-config", metavar="PATH",
        help="path to YAML formatted dataset config")
    p.add_argument(
        "--extra-config", metavar="PATH",
        help="path to YAML formatted extra config")

def validate_config_args(args):
    if args.pipeline_config and (
            args.model_config or
            args.train_config or
            args.eval_config or
            args.dataset_config):
        log.warning(
            "--pipeline-config specified, ignoring all "
            "other config options")

def init_config(args):
    if args.pipeline_config:
        return args.pipeline_config
    config = pipeline_pb2.TrainEvalPipelineConfig()
    _apply_model_config(args, config)
    _apply_train_config(args, config)
    _apply_eval_config(args, config)
    _apply_dataset_config(args, config)
    _apply_extra_config(args, config)
    _apply_arg_config(args, config)
    _write_config(config, CONFIG_FILENAME)
    return CONFIG_FILENAME

def _apply_model_config(args, config):
    if args.model_config:
        _apply_config(args.model_config, "model", config.model)

def _apply_train_config(args, config):
    if args.train_config:
        _apply_config(args.train_config, "train", config.train_config)

def _apply_eval_config(args, config):
    if args.eval_config:
        _apply_config(args.eval_config, "eval", config.eval_config)

def _apply_dataset_config(args, config):
    if args.dataset_config:
        _apply_config(args.dataset_config, "dataset", config)

def _apply_extra_config(args, config):
    if args.extra_config:
        _apply_config(args.extra_config, "extra", config)

def _apply_arg_config(args, config):
    _apply_config_arg(
        args, "train_steps",
        config.train_config, ["num_steps"])
    _apply_config_arg(
        args, "eval_examples",
        config.eval_config, ["num_examples"])

def _apply_config(src, desc, x):
    data = _try_load_config(src, desc)
    if not isinstance(data, dict):
        raise ConfigError(
            "invalid configuration in %s: expected dict, got %s"
            % (src, type(data)))
    _apply_dict(data, x)

def _try_load_config(src, desc):
    resolved = op_util.find_file(src)
    if not resolved:
        raise ConfigError("cannot find config %s" % src)
    log.info("Using %s config %s", desc, resolved)
    try:
        return yaml.load(open(resolved, "r"))
    except Exception as e:
        raise ConfigError("error reading config '%s': %s" % (resolved, e))

def _apply_dict(d, msg):
    msg.SetInParent()
    for name, val in d.items():
        if isinstance(val, dict):
            _apply_dict(val, getattr(msg, name))
        elif isinstance(val, list):
            _apply_list(val, getattr(msg, name))
        else:
            _set_attr(val, name, msg)

def _apply_list(l, msg):
    for item in l:
        if isinstance(item, dict):
            _apply_dict(item, msg.add())
        else:
            msg.append(item)

def _set_attr(val, name, msg):
    try:
        setattr(msg, name, val)
        _assert_real_val(msg, name, val)
    except TypeError:
        # try val as enum
        setattr(msg, name, getattr(msg, val))

def _assert_real_val(msg, name, expected):
    """Verify that attr name of msg equals expected.

    protobuf messages are subject to precision loss for floats if the
    C backend is used. This function raises an assertion error if the
    message attr value isn't equal to the expected value).
    """
    actual = getattr(msg, name)
    if actual != expected:
        raise AssertionError(
            "bad protobuf value for '%s': expected %s but got %s\n"
            "Try setting env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
            "fix this"
            % (name, expected, actual))

def _apply_config_arg(args, arg_name, msg, msg_attr_path):
    try:
        val = getattr(args, arg_name)
    except AttributeError:
        pass
    else:
        if val is None:
            return
        log.info(
            "Applying %s argument as %s=%s",
            arg_name, ".".join(msg_attr_path), val)
        setattr(
            _apply_attr_path(msg, msg_attr_path[:-1]),
            msg_attr_path[-1],
            val)

def _apply_attr_path(x, attr_path):
    for name in attr_path:
        x = getattr(x, name)
    return x

def _write_config(config, filename):
    with open(filename, "w") as f:
        f.write(text_format.MessageToString(config))
