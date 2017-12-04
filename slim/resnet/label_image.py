import argparse
import subprocess

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
    subprocess.check_call(cmd, shell=True)

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
