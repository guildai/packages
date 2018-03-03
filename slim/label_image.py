import argparse
import imghdr
import json
import re
import struct
import subprocess
import sys

def main():
    args = _parse_args()
    cmd_template = (
        "{} --graph={} --labels={} --output_layer={} "
        "--input_mean={} --input_std={} --image={} "
        "--input_width={} --input_height={}")
    image_width, image_height = _image_dim(args.image)
    print(
        "Using image dimensions width={}, height={}".format(
            image_width, image_height))
    cmd = cmd_template.format(
        args.label_image_bin,
        args.graph,
        args.labels,
        args.output_layer,
        args.input_mean,
        args.input_std,
        args.image,
        image_width,
        image_height,
    )
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.output)
        sys.exit(e.returncode)
    else:
        _write_prediction(_parse_prediction(out))
        sys.stderr.write(out)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-image-bin", default="./label_image")
    p.add_argument("--graph", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output-layer", required=True)
    p.add_argument("--input-mean", type=float, default=0.0)
    p.add_argument("--input-std", type=float, default=1.0)
    p.add_argument("--image", required=True)
    return p.parse_args()

def _image_dim(filename):
    # Adapted from http://bit.ly/2zW9vqD
    with open(filename, "rb") as f:
        head = f.read(24)
        assert len(head) == 24, filename
        image_type = imghdr.what(filename)
        if image_type == "png":
            check = struct.unpack(">i", head[4:8])[0]
            assert 0x0d0a1a0a, filename
            return struct.unpack(">ii", head[16:24])
        elif image_type == "gif":
            return struct.unpack("<HH", head[6:10])
        elif image_type == "jpeg":
            f.seek(0)
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf:
                f.seek(size, 1)
                byte = f.read(1)
                while ord(byte) == 0xff:
                    byte = f.read(1)
                ftype = ord(byte)
                size = struct.unpack(">H", f.read(2))[0] - 2
            f.seek(1, 1)  # Skip `precision" byte.
            height, width = struct.unpack(">HH", f.read(4))
            return width, height
        else:
            raise AssertionError(filename)

def _parse_prediction(out):
    m = re.findall(r"(\d+):([^ ]+?) \(\d+\): ([\.0-9]+)", out)
    return [[label, id, float(val)] for id, label, val in m]

def _write_prediction(vals):
    json.dump(vals, open("prediction.json", "w"))

if __name__ == "__main__":
    main()
