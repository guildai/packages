from __future__ import print_function

import argparse
import os
import sys

import yaml

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

# Ensure proper encoding of floats in protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format

from object_detection import model_main

from guild import op_util

CONFIG_FILENAME = "generated.config"

def main():
    args = _parse_args()
    _validate_args(args)
    config_path = _init_config(args)
    sys.argv = _model_main_argv(config_path, args.model_dir)
    try:
        tf.app.run(model_main.main)
    except KeyboardInterrupt:
        sys.stderr.write("Operation stopped by user\n")

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir", metavar="DIR",
        help="directory to write model checkpoints and logs")
    p.add_argument(
        "--pipeline-config", metavar="PATH",
        help=("path to pbtxt formatted detection config - "
              "overrides all other config"))
    p.add_argument(
        "--dataset-config", metavar="PATH",
        help="path to YAML formatted dataset config")
    p.add_argument(
        "--model-config", metavar="PATH",
        help="path to YAML formatted model config")
    p.add_argument(
        "--train-config", metavar="PATH",
        help="path to YAML formatted train config")
    p.add_argument(
        "--eval-config", metavar="PATH",
        help="path to YAML formatted eval config")
    return p.parse_args()

def _validate_args(args):
    if args.pipeline_config and (
            args.dataset_config or
            args.model_config or
            args.train_config or
            args.eval_config):
        sys.stderr.write(
            "--pipeline-config specified, ignoring all other config options\n")

def _init_config(args):
    if args.pipeline_config:
        return args.pipeline_config
    return _generate_config(
        args.dataset_config,
        args.model_config,
        args.train_config,
        args.eval_config)

def _generate_config(dataset_src, model_src, train_src, eval_src):
    from object_detection.protos import pipeline_pb2
    config = pipeline_pb2.TrainEvalPipelineConfig()
    if model_src:
        _apply_config(model_src, "model", config.model)
    if train_src:
        _apply_config(train_src, "train", config.train_config)
    if eval_src:
        _apply_config(eval_src, "eval", config.eval_config)
    if dataset_src:
        _apply_config(dataset_src, "dataset", config)
    _write_config(config, CONFIG_FILENAME)
    return CONFIG_FILENAME

def _apply_config(src, desc, x):
    data = _try_load_config(src, desc)
    if not isinstance(data, dict):
        _error(
            "invalid configuration in %s: expected dict, got %s"
            % (src, type(data)))
    _apply_dict(data, x)

def _try_load_config(src, desc):
    resolved = op_util.find_file(src)
    if not resolved:
        _error("cannot find config %s" % src)
    _info("Using %s config %s" % (desc, resolved))
    try:
        return yaml.load(open(resolved, "r"))
    except Exception as e:
        _error("error reading config '%s': %s" % (resolved, e))

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
        if getattr(msg, name) != val:
            import pdb;pdb.set_trace()
    except TypeError:
        # try val as enum
        setattr(msg, name, getattr(msg, val))

def _write_config(config, filename):
    with open(filename, "w") as f:
        f.write(text_format.MessageToString(config))

def _model_main_argv(config_path, model_dir):
    argv = [sys.argv[0]]
    argv.extend(["--pipeline_config_path", config_path])
    if model_dir:
        argv.extend(["--model_dir", model_dir])
    return argv

def _info(msg):
    sys.stderr.write(msg)
    sys.stderr.write("\n")

def _error(msg):
    _info("train: %s" % msg)
    sys.exit(1)

if __name__ == "__main__":
    main()
