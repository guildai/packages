from __future__ import print_function

import argparse
import glob
import logging
import os

# Ensure proper encoding of floats in protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format

from object_detection.protos import pipeline_pb2

log = logging.getLogger()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s")
    args = _parse_args()
    _factor_all(args)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("in_dir", metavar="IN_DIR")
    p.add_argument("out_dir", metavar="OUT_DIR")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def _factor_all(args):
    for path in glob.glob(os.path.join(args.in_dir, "*.config")):
        config = _try_load_config(path, args)
        if config is None:
            log.warn("unable to load %s, skipping", path)
        _factor_config(config, path, args)

def _try_load_config(path, args):
    try:
        return _load_config(path)
    except Exception:
        if args.debug:
            log.exception("loading %s" % path)
        return None

def _load_config(path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(path, "r") as f:
        text_format.Parse(f.read(), config)
    return config

def _factor_config(config, src, args):
    src_name = os.path.basename(src)
    _factor_attr("model", config, src_name, args.out_dir)
    _factor_attr("train_config", config, src_name, args.out_dir)
    _factor_attr("train_input_reader", config, src_name, args.out_dir)
    _factor_attr("eval_config", config, src_name, args.out_dir)
    _factor_attr("eval_input_reader", config, src_name, args.out_dir)

def _factor_attr(attr_name, config, config_name, out_dir):
    attr_val = getattr(config, attr_name)
    config_path = os.path.join(out_dir, attr_name, config_name)
    _ensure_dir(config_path)
    with open(config_path, "w") as f:
        f.write(text_format.MessageToString(attr_val))

def _ensure_dir(filename):
    dir = os.path.dirname(filename)
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != 17:
            raise

if __name__ == "__main__":
    main()
