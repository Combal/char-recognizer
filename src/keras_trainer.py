import src.vnist as vnist
from src.network import init_model
from keras.callbacks import TensorBoard

nb_classes = 39
nb_epoch = 24
batch_size = 128
img_rows, img_cols = 56, 56

dataset = vnist.read_data_sets('data')
X_train = dataset.train.images
Y_train = dataset.train.labels
X_validation = dataset.validation.images
Y_validation = dataset.validation.labels
X_test = dataset.test.images
Y_test = dataset.test.labels

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_validation.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')
print('Y_train shape:', Y_train.shape)

model = init_model(nb_classes=nb_classes, input_shape=input_shape)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
        verbose=1, validation_split=0.1, callbacks=[callback])

score = model.evaluate(X_test, Y_test, verbose=0)
print('test score: ', score[0])
print('test accuracy: ', score[1])
model.save_weights('data/model.h5')
