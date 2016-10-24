#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import image_reader as ir
import numpy as np

DIR = "../data/categories"


class DataInitializer:

	def __init__(self):
		# self._data = []
		self._training_images = []
		self._training_labels = []
		self._validation_images = []
		self._validation_labels = []
		self._test_images = []
		self._test_labels = []
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._list_eye = lambda n: np.eye(n).tolist()
		self._y = self._list_eye(33)
		self._labels = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
		self.load_data()
		self._num_examples = len(self._training_labels)
		self._validation_size = self._num_examples * 0.2

	def get_label_vector(self, l):
		if l in self._labels:
			return self._y[self._labels.index(l) / 3]
		return None

	def load_data(self):
		print "loading images..."
		categories = os.listdir(DIR)
		for label in categories:
			# print os.path.join(DIR, categories[i])
			label_dir = os.path.join(DIR, label)
			if not os.path.isdir(label_dir):
				continue
			# print label, get_label_vector(label)
			images = os.listdir(label_dir)
			for image in images:
				image_path = os.path.join(label_dir, image)
				# print image_path
				image_rec = ir.read_and_transform(image_path)
				self._training_images.append(image_rec)
				self._training_labels.append(self.get_label_vector(label))
				# self._data.append((image_rec, label, self.get_label_vector(label)))
		self._training_images = np.array(self._training_images)
		self._training_labels = np.array(self._training_labels)
		print "images has been loaded"

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1
			print "start new epoch: %s" % self._epochs_completed
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._training_images = self._training_images[perm]
			self._training_labels = self._training_labels[perm]
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._training_images[start:end]

	def get_epochs(self):
		return self._epochs_completed

