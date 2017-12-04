import argparse
import json
import re
import subprocess
import sys

def main():
    args = _parse_args()
    cmd_template = (
        "{} --graph={} --labels={} --output_layer={} "
        "--input_mean=0.000000 --input_std=3 --image={}")
    cmd = cmd_template.format(
        args.label_image_bin,
        args.graph,
        args.labels,
        args.output_layer,
        args.image
    )
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.output)
        sys.exit(e.returncode)
    else:
        _write_prediction(_parse_prediction(out))
        sys.stderr.write(out)

def _parse_prediction(out):
    m = re.findall(r"(\d+):([^ ]+?) \(\d+\): ([\.0-9]+)", out)
    return [[label, id, float(val)] for id, label, val in m]

def _write_prediction(vals):
    json.dump(vals, open("prediction.json", "w"))

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-image-bin", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output-layer", required=True)
    p.add_argument("--image", required=True)
    return p.parse_args()

if __name__ == "__main__":
    main()
