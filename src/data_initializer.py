#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import image_reader as ir
import glob
import numpy as np

DIR = "../data/categories"
list_eye = lambda n: np.eye(n).tolist()

y = list_eye(33)

labels = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"


def get_label_vector(l):
	if l in labels:
		return y[labels.index(l)/3]
	return None


def get_data():
	data = []
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
		data.append((image_rec, label, get_label_vector(label)))

	return data

