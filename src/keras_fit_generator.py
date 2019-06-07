from src.network import init_model
from keras.callbacks import TensorBoard
import time
from src.dataset import train_generator, valid_generator, test_generator

nb_classes = 39
nb_epoch = 30
img_rows, img_cols = 56, 56

input_shape = (img_rows, img_cols, 1)

model = init_model(nb_classes=nb_classes, input_shape=input_shape)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model.fit_generator(train_generator,
                    validation_data=valid_generator,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_steps=100,
                    steps_per_epoch=100,
                    callbacks=[callback])

model.save_weights('data/models/model-{}.h5'.format(time.time()))

score = model.evaluate_generator(test_generator, steps=100, verbose=0)
print('test score: {}, accuracy: {}'.format(score[0], score[1]))
