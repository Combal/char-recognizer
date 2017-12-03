from cv2 import cv2
import numpy as np
import math
import copy
import sys

FILE = "data/char.jpg"
MOD_FILE = "data/char-mod.jpg"
IMAGE_SIZE = 56
N_INPUT = IMAGE_SIZE * IMAGE_SIZE
IMAGE_PADDING = int(round(IMAGE_SIZE * 0.2))


def get_global_bounding_rect(contours):
    min_x = sys.maxint
    min_y = sys.maxint
    max_w = 0
    max_h = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_w:
            max_w = x + w
        if y + h > max_h:
            max_h = y + h
    return min_x, min_y, max_w, max_h


def crop_image(img):
    img2 = copy.copy(img)
    image, contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x1, y1, x2, y2 = get_global_bounding_rect(contours)
    del img2
    return img[y1:y2, x1:x2]


def read_and_transform(image_path=FILE):
    if image_path is None:
        image_path = FILE
    # image_rec = np.zeros((1, N_INPUT))

    # read file in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # invert colors and resize to 28x28
    # img = cv2.resize(255 - img, (28, 28))
    img = 255 - img

    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    img = crop_image(img)

    rows, cols = img.shape

    if rows == 0:
        print("skipping image %s" % image_path)
        return None

    if rows > cols:
        factor = (IMAGE_SIZE - IMAGE_PADDING) / (rows * 1.0)
        rows = IMAGE_SIZE - IMAGE_PADDING
        cols = int(round(cols * factor))
        # first cols than rows
        img = cv2.resize(img, (cols, rows))
    else:
        factor = (IMAGE_SIZE - IMAGE_PADDING) / (cols * 1.0)
        cols = IMAGE_SIZE - IMAGE_PADDING
        rows = int(round(rows * factor))
        # first cols than rows
        img = cv2.resize(img, (cols, rows))
    cols_padding = (int(math.ceil((IMAGE_SIZE - cols) / 2.0)), int(math.floor((IMAGE_SIZE - cols) / 2.0)))
    rows_padding = (int(math.ceil((IMAGE_SIZE - rows) / 2.0)), int(math.floor((IMAGE_SIZE - rows) / 2.0)))
    img = np.lib.pad(img, (rows_padding, cols_padding), 'constant')

    # cv2.imwrite(MOD_FILE, img)
    # shiftx, shifty = get_best_shift(img)
    # shifted = shift(img, shiftx, shifty)
    # img = shifted

    # save modified image

    # scale pixel values into range [0, 1]
    # flatten = img.flatten() / 255.0
    # flatten[flatten < 0.5] = 0
    # flatten[flatten >= 0.5] = 1
    # # print flatten
    # image_rec[0] = flatten
    # # print image_rec
    # return flatten
    return img

# print i

if __name__ == '__main__':
    img = read_and_transform()
    cv2.imwrite(MOD_FILE, img)
