from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers

nb_filters = 64
pool_size = (2, 2)
kernel_size = (3, 3)


def init_model(nb_classes, input_shape):
    model = Sequential()
    
    model.add(Convolution2D(32, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(32, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # model.add(Convolution2D(nb_filters, kernel_size, use_bias=True))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    # model.add(Dense(256, kernel_regularizer=regularizers.l2(0.1)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
