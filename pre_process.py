import pandas as pd
import numpy as np
import logging
from scipy import misc

def save_csv_data_to_npy():
    data = pd.read_csv('data/train.csv')
    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]
    logging.info('train_data.shape:{} train_label.shape:{}'.format(
        train_data.shape, train_label.shape))
    np.save('data/train_data.npy', train_data)
    np.save('data/train_label.npy', train_label)

    data = pd.read_csv('data/test.csv')
    test_data = data.values[0:, 0:]
    logging.info('test_data.shape:{}'.format(test_data.shape))
    np.save('data/test_data.npy', test_data)


def resize_data():
    def process(source_path, result_path, rate=0.5):
        data = np.load(source_path)
        tmp = [None for _ in data]
        for i, img in enumerate(data):
            img = img.reshape((28,28))
            img = misc.imresize(img, rate)
            tmp[i] = img.flatten()
        data = np.array(tmp)
        np.save(result_path, data)
    
    rate = 0.5
    source_path = 'data/train_data.npy'
    result_path = 'data/train_data_resized.npy'
    process(source_path, result_path, rate)

    source_path = 'data/test_data.npy'
    result_path = 'data/test_data_resized.npy'
    process(source_path, result_path, rate)


def main(args):
    command = args['command']
    eval('{}()'.format(command))
