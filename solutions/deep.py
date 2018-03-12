import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import keras
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,  ZeroPadding2D, Input, BatchNormalization
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils
import functions


def train_for_model(flags):
    data = np.load('data/' + flags['train_data'])
    label = np.load('data/' + flags['train_label'])
    test_data = np.load('data/' + flags['test_data'])

    batch_size = 128
    nb_classes = 10
    epochs = 1000

    # input image dimensions
    img_rows, img_cols = 14, 14
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    
    x_train = data.reshape(data.shape[0], 14, 14, 1)
    x_test = test_data.reshape(test_data.shape[0], 14, 14, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    y_train = np_utils.to_categorical(label, nb_classes)

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    test_result=model.predict_classes(x_test, batch_size=128, verbose=1)
    result = np.c_[range(1,len(test_result)+1), test_result.astype(int)]
    df_result = pd.DataFrame(result[:,0:2], columns=['ImageId', 'Label'])

    df_result.to_csv('data/results_dl.csv', index=False)

def predict(flags):
    # model_path = 'data/' + flags['train_model']
    # model = joblib.load(model_path)    
    # data_path = 'data/' + flags['test_data']
    # data = np.load(data_path)
    # label = model.predict(data)
    # functions.save_result(label, 'xxxx')
    pass


def main(flags):
    logging.info('start!!!!!!')
    if flags['command'] == 'train':
        train_for_model(flags)
    elif flags['command'] == 'predict':
        predict(flags)
    else:
        logging.error('illegal command!')
    logging.info('end!!!!!!')


# command
# train
# predict