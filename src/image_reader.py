import cv2
import numpy as np
import math
from scipy import ndimage

FILE = "../data/char.jpg"
MOD_FILE = "../data/char-mod.jpg"
N_INPUT = 2500


# Reads and converts file for recognition
def get_best_shift(img):
	cy, cx = ndimage.measurements.center_of_mass(img)

	rows, cols = img.shape
	shiftx = np.round(cols / 2.0 - cx).astype(int)
	shifty = np.round(rows / 2.0 - cy).astype(int)

	return shiftx, shifty


# Shifts image in one pixel
def shift(img, sx, sy):
	rows, cols = img.shape
	M = np.float32([[1, 0, sx], [0, 1, sy]])
	shifted = cv2.warpAffine(img, M, (cols, rows))

	return shifted


def read_and_transform(image_path=FILE):
	if image_path is None:
		image_path = FILE
	image_rec = np.zeros((1, N_INPUT))

	# read file in grayscale mode
	img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

	# invert colors and resize to 28x28
	# img = cv2.resize(255 - img, (28, 28))
	img = 255 - img

	# leave only 0-s and 255-s
	(thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
	# img[img < 10] = 0
	# blur = cv2.GaussianBlur(img, (5, 5), 0)
	# (ret3, img) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	while np.sum(img[0]) == 0:
		img = img[1:]

	while np.sum(img[:, 0]) == 0:
		img = np.delete(img, 0, 1)

	while np.sum(img[-1]) == 0:
		img = img[:-1]

	while np.sum(img[:, -1]) == 0:
		img = np.delete(img, -1, 1)

	rows, cols = img.shape

	if rows > cols:
		factor = 40.0 / rows
		rows = 40
		cols = int(round(cols * factor))
		# first cols than rows
		img = cv2.resize(img, (cols, rows))
	else:
		factor = 40.0 / cols
		cols = 40
		rows = int(round(rows * factor))
		# first cols than rows
		img = cv2.resize(img, (cols, rows))
	cols_padding = (int(math.ceil((50 - cols) / 2.0)), int(math.floor((50 - cols) / 2.0)))
	rows_padding = (int(math.ceil((50 - rows) / 2.0)), int(math.floor((50 - rows) / 2.0)))
	img = np.lib.pad(img, (rows_padding, cols_padding), 'constant')

	# shiftx, shifty = get_best_shift(img)
	# shifted = shift(img, shiftx, shifty)
	# img = shifted

	# save modified image
	# cv2.imwrite(MOD_FILE, img)

	# scale pixel values into range [0, 1]
	flatten = img.flatten() / 255.0
	flatten[flatten < 0.5] = 0
	flatten[flatten >= 0.5] = 1
	# print flatten
	image_rec[0] = flatten
	# print image_rec
	return flatten


# print i

if __name__ == '__main__':
	read_and_transform()
