from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from src.vnist import labels_string

BATCH_SIZE = 32
TARGET_SHAPE = (56, 56)


def f(img):
    # img = transform(img.reshape)
    # print(img.shape, flush=True)
    return 255 - img


datagen = ImageDataGenerator(
    rotation_range=10,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    rescale=1. / 255.0,
    # zca_whitening=True,
    # samplewise_center=True,
    # featurewise_center=True,
    preprocessing_function=f,
    fill_mode='nearest')

classes = [char for char in labels_string]

train_generator = datagen.flow_from_directory(directory=r'./data/train_images',
                                              target_size=TARGET_SHAPE,
                                              color_mode='grayscale',
                                              batch_size=BATCH_SIZE,
                                              classes=classes,
                                              shuffle=True
                                              # save_to_dir='data/preview',
                                              # save_prefix='cat',
                                              # save_format='png'
                                              )

valid_generator = datagen.flow_from_directory(directory=r'./data/train_images',
                                              target_size=TARGET_SHAPE,
                                              color_mode='grayscale',
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              classes=classes)

test_generator = datagen.flow_from_directory(directory=r'./data/train_images',
                                             target_size=TARGET_SHAPE,
                                             color_mode='grayscale',
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             classes=classes)

if __name__ == '__main__':
    BATCH_SIZE = 1
    i = 0
    for x, y in train_generator:
        i += 1
        label = classes[int(np.argmax(y[0]))]
        print(label)
        cv2.imshow(label, x[0])
        if i > 5:
            break

    cv2.waitKey(0)
