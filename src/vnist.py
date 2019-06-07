#!/usr/bin/env python
import os
import pickle

import cv2
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base

from src import image_utils

DATASET_FILE = 'data/dataset.pkl'
VALIDATION_RATE = 0
TEST_RATE = 0.20


y = np.eye(39).tolist()
labels_string = "აბგდdეeვvზთიკლlმნოoპჟრrსტუფქღყშჩცძწჭხჯჰ"


def get_label_vector(l):
    return y[labels_string.index(l)]


def get_label(pred):
    char = labels_string[pred]
    if char == 'd':
        char = 'დ'
    elif char == 'e':
        char = 'ე'
    elif char == 'v':
        char = 'ვ'
    elif char == 'l':
        char = 'ლ'
    elif char == 'o':
        char = 'ო'
    elif char == 'r':
        char = 'რ'
    return char


def read_data_sets(train_dir):
    print("loading images...")
    train_images, train_labels = read_from_folder(os.path.join(train_dir, "train_images"))
    validation_size = int(round(len(train_images) * VALIDATION_RATE))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    test_size = int(round(len(train_images) * TEST_RATE))
    test_images = train_images[:test_size]
    test_labels = train_labels[:test_size]
    train_images = train_images[test_size:]
    train_labels = train_labels[test_size:]
    # test_images, test_labels = read_from_folder(os.path.join(train_dir, "test_images"))

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    print("images has been loaded")
    return base.Datasets(train=train, validation=validation, test=test)


def read_from_folder(folder):
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'rb') as f:
            return pickle.load(f)
    if os.path.exists(folder):
        categories = os.listdir(folder)
        images = []
        labels = []
        for label in categories:
            label_dir = os.path.join(folder, label)
            if not os.path.isdir(label_dir):
                continue
            image_list = os.listdir(label_dir)
            for image in image_list:
                image_path = os.path.join(label_dir, image)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_rec = image_utils.transform(img)
                if image_rec is None:
                    print("skipping image {}".format(image_path))
                    continue
                images.append(image_rec)
                labels.append(get_label_vector(label))

        images = np.array(images)
        labels = np.array(labels)
        with open(DATASET_FILE, 'wb') as f:
            pickle.dump((images, labels), f)
        return images, labels
    import urllib.request
    urllib.request.urlretrieve('https://drive.google.com/uc?id=1-AFAp5UcIiuKlL9YCxa1xhiOLSNq8X7O&export=download',
                               DATASET_FILE)
    with open(DATASET_FILE, 'rb') as f:
        return pickle.load(f)


class DataSet:
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], \
            ('images.shape: {} labels.shape: {}'.format(images.shape, labels.shape))
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


if __name__ == '__main__':
    i, l = read_from_folder('data/train_images')
    print(i.shape)
    print(l.shape)
