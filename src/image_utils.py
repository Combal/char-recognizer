from cv2 import cv2
import numpy as np
import math
import sys

IMAGE_SIZE = 56
N_INPUT = IMAGE_SIZE * IMAGE_SIZE
IMAGE_PADDING = int(round(IMAGE_SIZE * 0.2))


def get_global_bounding_rect(contours):
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    boxes = np.asarray(boxes)
    left = np.min(boxes[:, 0])
    top = np.min(boxes[:, 1])
    right = np.max(boxes[:, 2])
    bottom = np.max(boxes[:, 3])
    return left, top, right, bottom


def crop_image(img):
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x1, y1, x2, y2 = get_global_bounding_rect(contours)
    return img[y1:y2, x1:x2]


def transform(img):
    img = 255 - img
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img = crop_image(img)
    return get_squared_with_padding(img)


def get_squared_with_padding(img):
    rows, cols = img.shape
    if rows == 0:
        return None

    a, b = (rows, cols) if rows > cols else (cols, rows)
    factor = (IMAGE_SIZE - IMAGE_PADDING) / (a * 1.0)
    a = IMAGE_SIZE - IMAGE_PADDING
    b = int(round(b * factor))
    img = cv2.resize(img, (b, a) if rows > cols else (a, b))

    a_padding = (int(math.ceil((IMAGE_SIZE - b) / 2.0)), int(math.floor((IMAGE_SIZE - b) / 2.0)))
    b_padding = (int(math.ceil((IMAGE_SIZE - a) / 2.0)), int(math.floor((IMAGE_SIZE - a) / 2.0)))
    rows_padding, cols_padding = (b_padding, a_padding) if rows > cols else (a_padding, b_padding)
    img = np.lib.pad(img, (rows_padding, cols_padding), 'constant')
    return img


if __name__ == '__main__':
    file = sys.argv[1] if len(sys.argv) > 1 else 'data/temp_file.jpg'
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original image', img)
    cv2.waitKey(0)
    img = transform(img)
    cv2.imshow('processed image', img)
    cv2.waitKey(0)
