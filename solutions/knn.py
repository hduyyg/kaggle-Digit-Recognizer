import logging
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import functions


def get_knn_classify(trainData, trainLabel):
    logging.info('start get knnClf by train.')
    knnClf = KNeighborsClassifier(n_neighbors=3)
    knnClf.fit(trainData, np.ravel(trainLabel))  # ravel 
    return knnClf


def predict(train_data, train_label, test_data):
    knn_clf = get_knn_classify(train_data, train_label)
    joblib.dump(knn_clf, 'data/knn.m')
    logging.info('start predict the result.')
    test_label = knn_clf.predict(test_data)
    functions.save_result(test_label, 'result_knn')


def test_args(train_data, train_label):
    pass


def main(flags):
    logging.info('start the knn!!!')
    train_data = np.load('data/train_data_01.npy')
    train_label = np.load('data/train_label.npy')
    
    if flags['command'] == 'predict':
        test_data = np.load('data/test_data_01.npy')
        predict(train_data, train_label, test_data)
    elif flags['command'] == 'test_args':
        test_args(train_data, train_label)
    else:
        logging.error('no this command: {}'.format(flags['command']))
    logging.info('complete the knn!!!')
