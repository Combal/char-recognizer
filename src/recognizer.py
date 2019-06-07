#!/usr/bin/env python

import argparse

import numpy as np

import src.vnist as vnist
from src.network import init_model

MODEL = 'data/models/model-1559506571.3370159.h5'
nb_classes = 39
input_shape = (56, 56, 1)


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
        self.model.load_weights(MODEL)
        # model.summary()

    def recognize(self, img):
        array = np.asarray(img, dtype=np.float32)

        array /= 255.0
        array = array.reshape(input_shape)
        array = np.expand_dims(array, 0)
        # print(array.shape)

        pred = self.model.predict_classes(array, batch_size=1, verbose=0)
        char = vnist.get_label(pred[0])

        print("{} - {}".format(pred[0], char), flush=True)
        return char


if __name__ == '__main__':
    #     array = 255 - array
    args = init_arguments()
    recognizer = Recognizer()
    recognizer.recognize(args.image)


