import pandas as pd
import numpy as np
import logging
from scipy import misc
from sklearn.decomposition import PCA

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


# def change_data_to_01():
#     data = np.load('data/train_data_resized.npy')
#     data = np.array(data != np.zeros_like(data), dtype=int)
#     logging.info('start to save train_data_resized_01!!!')
#     np.save('data/train_data_resized_01.npy', data)

#     data = np.load('data/test_data_resized.npy')
#     data = np.array(data != np.zeros_like(data), dtype=int)
#     logging.info('start to save test_data_resized_01!!!')
#     np.save('data/test_data_resized_01.npy', data)


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


def resize_pca_data():
    def process(source_path, result_path):
        data = np.load(source_path)
        model = PCA(n_components='mle')
        model = model.fit(data)
        new_data = model.transform(data)
        logging.info('pca:souce:{} res:{}'.format(data.shape, new_data.shape))
        np.save(new_data, result_path)


    source_path = 'data/train_data_resized.npy'
    result_path = 'data/train_data_resized_pca.npy'
    process(source_path, result_path)

    source_path = 'data/test_data_resized.npy'
    result_path = 'data/test_data_resized_pca.npy'
    process(source_path, result_path)


def main(args):
    command = args['command']
    eval('{}()'.format(command))
