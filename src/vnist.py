#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import src.image_reader as ir
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
import pickle

DATASET_FILE = 'data/dataset.pkl'
VALIDATION_RATE = 0
TEST_RATE = 0.20


def list_eye(n):
    return np.eye(n).tolist()


y = list_eye(39)
# labels_string = "აბგ"
labels_string = u"აბგდdეeვvზთიკლlმნოoპჟრrსტუფქღყშჩცძწჭხჯჰ"


def get_label_vector(l):
    # if l in labels_string:
    return y[labels_string.index(l)]
    # return None


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
            # print os.path.join(DIR, categories[i])
            label_dir = os.path.join(folder, label)
            if not os.path.isdir(label_dir):
                continue
            # print label, get_label_vector(label)
            image_list = os.listdir(label_dir)
            for image in image_list:
                image_path = os.path.join(label_dir, image)
                # print image_path
                image_rec = ir.read_and_transform(image_path)
                if image_rec is None:
                    continue
                images.append(image_rec)
                labels.append(get_label_vector(label))
            # print label
            # print get_label_vector(label)
            # self._data.append((image_rec, label, self.get_label_vector(label)))
        images = np.array(images)
        labels = np.array(labels)
        with open(DATASET_FILE, 'wb') as f:
            pickle.dump((images, labels), f)
        return images, labels
    import urllib.request
    urllib.request.urlretrieve('https://drive.google.com/uc?id=1-AFAp5UcIiuKlL9YCxa1xhiOLSNq8X7O&export=download', DATASET_FILE)
    with open(DATASET_FILE, 'rb') as f:
        return pickle.load(f)


class DataSet:
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)
        )
        self._num_examples = images.shape[0]
        # assert images.shape[3] == 1
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
