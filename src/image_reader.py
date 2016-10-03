import cv2
import numpy as np

FILE = "../data/char.jpeg"
MOD_FILE = "../data/char-mod.jpeg"
N_INPUT = 784

def read_and_transform():

	image_rec = np.zeros((1, N_INPUT))

	# read file in grayscale mode
	i = cv2.imread(FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)

	# invert colors and resize to 28x28
	i = cv2.resize(255 - i, (28, 28))

	# leave only 0-s and 255-s
	(thresh, i) = cv2.threshold(i, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# save modified image
	cv2.imwrite(MOD_FILE, i)

	# scale pixel values into range [0, 1]
	flatten = i.flatten() / 255.0

	# print flatten
	image_rec[0] = flatten
	print image_rec
	return flatten


# print i

if __name__ == '__main__':
	read_and_transform()
