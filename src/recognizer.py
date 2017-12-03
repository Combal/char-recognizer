#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from PIL import Image
import argparse

import numpy as np
import src.vnist as vnist
from src.network import init_model

nb_classes = 39
input_shape = [56, 56, 1]


def init_arguments():
    parser = argparse.ArgumentParser(description='Georgian OCR')
    parser.add_argument('-i', '--image', metavar='image_path', type=str,
                        help='Path to the image to recognize.')
    # parser.add_argument('-W', '--weights', metavar='weights_path', type=str,
    #                     help='Path to the weights.')
    # parser.add_argument('-w', '--width', metavar='image_width', type=int,
    #                     help='image width: 128 / 256 / 512 (256 is default)', default=256)
    # parser.add_argument('-m', '--model', metavar='model', type=str,
    #                     help='Path to model')
    # parser.add_argument('-e', '--english', action='store_true',
    #                     help='print output in english letters')
    return parser.parse_args()


class Recognizer:

    def __init__(self):
        self.model = init_model(nb_classes, input_shape=input_shape)
        # model.summary()
        self.model.load_weights('data/model.h5')

    def recognize(self, image_path):
        img = Image.open(image_path)
        img = img.convert("L")
        array = np.asarray(img.getdata(), dtype=np.float32)
        if __name__ == '__main__':
            array = 255 - array
        array /= 255.0
        array = array.reshape(input_shape)
        # print(array.shape)

        array = np.expand_dims(array, 0)

        pred = self.model.predict_classes(array, batch_size=1, verbose=0)
        char = vnist.labels_string[pred[0]]
        if char == 'd':
            char = u'დ'
        elif char == 'e':
            char = u'ე'
        elif char == 'v':
            char = u'ვ'
        elif char == 'l':
            char = u'ლ'
        elif char == 'o':
            char = u'ო'
        elif char == 'r':
            char = u'რ'
        print("%d - %s" % (pred[0], char))
        return char

if __name__ == '__main__':
    args = init_arguments()
    recognizer = Recognizer()
    recognizer.recognize(args.image)


