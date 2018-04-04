from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

import skimage.io

import coco
import model as modellib
import visualize

class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

def main():
    args = init_args()
    model = init_model(args)
    detect(args.images, model)

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="logs")
    p.add_argument("--model-weights", default="mask_rcnn_coco.h5")
    p.add_argument("--images", required=True)
    return p.parse_args()

def init_model(args):
    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    model = modellib.MaskRCNN(
        mode="inference",
        model_dir=args.model_dir,
        config=config)
    model.load_weights(args.model_weights, by_name=True)
    return model

def init_config():
    return config

def detect(images, model):
    for image_path in glob.glob(images):
        image = skimage.io.imread(image_path)
        results = model.detect([image], verbose=1)
        r = results[0]
        import pdb;pdb.set_trace()
        visualize.display_instances(
            image,
            r['rois'],
            r['masks'],
            r['class_ids'],
            class_names,
            r['scores'])

if __name__ == "__main__":
    main()
