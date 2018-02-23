# pylint: disable=missing-docstring,too-many-locals,invalid-name
# pylint: disable=too-few-public-methods,too-many-arguments

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import keras
import numpy as np
import pandas as pd

from PIL import Image

input_shape = [64, 64, 1]

class State(object):

    def __init__(self, **kw):
        for name, val in kw.items():
            setattr(self, name, val)

def generate():
    args = _parse_args()
    img, img_width, img_height = _init_image(args)
    img_base, _ = os.path.splitext(os.path.basename(args.image))
    char_list = _init_char_list(args)
    state = State(
        img=img,
        img_width=img_width,
        img_height=img_height,
        img_base=img_base,
        model=_init_model(args),
        char_list=char_list,
        char_dict=_init_char_dict(args),
        space=_init_space(char_list),
        output=args.output
    )
    for slide in range(args.slides):
        _generate_slide(slide, state)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slides", default=5, type=int)
    p.add_argument("--image", required=True)
    p.add_argument("--image-width", default=0, type=int)
    p.add_argument("--model", default="model/model.json")
    p.add_argument("--weights", default="model/weights.hdf5")
    p.add_argument("--char-list", default="data/char_list.csv")
    p.add_argument("--char-dict", default="data/char_dict.pkl")
    p.add_argument("--output", default="output")
    return p.parse_args()

def _init_image(args):
    img = Image.open(args.image)
    orig_width, _ = img.size
    if args.image_width == 0:
        new_width = orig_width
    else:
        new_width = args.image_width
    new_height = int(img.size[1] * new_width / img.size[0])
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img = np.array(img)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    margin = (input_shape[0] - 18) // 2
    img_new = np.ones(
        [img.shape[0] + 2 * margin + 18,
         img.shape[1] + 2 * margin + 18],
        dtype=np.uint8) * 255
    img_new[margin:margin + new_height, margin:margin + new_width] = img
    img = img_new.astype(np.float32) / 255 # pylint: disable=no-member
    return img, new_width, new_height

def _init_model(args):
    model_json = open(args.model).read()
    model = keras.models.model_from_json(model_json)
    model.load_weights(args.weights)
    return model

def _init_char_list(args):
    char_list = pd.read_csv(args.char_list, encoding="cp932")
    char_list = char_list[char_list['frequency'] >= 10]
    return char_list['char'].as_matrix()

def _init_space(char_list):
    for k, v in enumerate(char_list):
        if v == " ":
            return k
    raise AssertionError()

def _init_char_dict(args):
    with open(args.char_dict, mode='rb') as f:
        return pickle.load(f)

def _generate_slide(slide, state):
    print("Generating slide %i" % slide)
    num_line = (state.img.shape[0] - input_shape[0]) // 18
    img_width = state.img.shape[1]
    new_line = np.ones([1, img_width])
    state.img = np.concatenate([new_line, state.img], axis=0)
    predicts = []
    text = []

    for h in range(num_line):
        w = 0
        penalty = 1
        predict_line = []
        text_line = ""
        while w <= img_width - input_shape[1]:
            input_img = state.img[h * 18:h * 18 + input_shape[0],
                                  w:w + input_shape[1]]
            input_img = input_img.reshape(
                [1, input_shape[0], input_shape[1], 1])
            predict = state.model.predict(input_img)
            if penalty:
                predict[0, state.space] = 0
            predict = np.argmax(predict[0])
            penalty = predict == state.space
            char = state.char_list[predict]
            predict_line.append(char)
            char_width = state.char_dict[char].shape[1]
            w += char_width
            text_line += char
        predicts.append(predict_line)
        text.append(text_line + '\r\n')

    img_aa = np.ones_like(state.img, dtype=np.uint8) * 255
    for h in range(num_line):
        w = 0
        for char in predicts[h]:
            char_width = state.char_dict[char].shape[1]
            char_img = 255 - state.char_dict[char].astype(np.uint8) * 255
            img_aa[h * 18:h * 18 + 16, w:w + char_width] = char_img
            w += char_width
    img_aa = Image.fromarray(img_aa)
    img_aa = img_aa.crop([0, slide, state.img_width, state.img_height + slide])

    _write_slide(slide, img_aa, text, state)

def _write_slide(slide, img, text, state):
    if not os.path.exists(state.output):
        os.makedirs(state.output)
    name_base = "%s_w%i_slide%i" % (state.img_base, state.img_width, slide)
    img_path = os.path.join(state.output, name_base + ".png")
    img.save(img_path)
    text_path = os.path.join(state.output, name_base + ".txt")
    with open(text_path, 'wb') as f:
        for line in text:
            f.write(line.encode("utf-8"))

if __name__ == "__main__":
    generate()
