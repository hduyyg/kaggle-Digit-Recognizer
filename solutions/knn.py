import logging
import pandas
import numpy
import sklearn
from sklearn import neighbors, cross_validation
from sklearn.neighbors import KNeighborsClassifier


logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename='digit_recognize_knn.log',
        filemode='w+')


def get_data():
    data = pandas.read_csv('../data/train.csv')
    data1 = pandas.read_csv('../data/test.csv')

    train_data = data.values[0:, 1:]
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]
    return train_data, train_label, test_data


def save_result(data, file_name):
    logging.info('start save result to {}!'.format(file_name))
    info = {}
    info['ImageId'] = [i for i in range(1, len(data) + 1)]
    info['Label'] = data
    data_frame = pandas.DataFrame(info)
    data_frame.to_csv(file_name, index=False, sep=',')


def knn_classify(trainData, trainLabel):
    knnClf = neighbors.KNeighborsClassifier(n_neighbors=3)
    knnClf.fit(trainData, numpy.ravel(trainLabel))  # ravel 
    return knnClf


def main():
    logging.info('start the knn!!!')
    trainData, trainLabel, testData = get_data()
    knnClf = knn_classify(trainData, trainLabel)
    testLabel = knnClf.predict(testData)
    save_result(testLabel, '../data/result_sklearn_knn.csv')
    logging.info('complete the knn!!!')


if __name__ == '__main__':
    main()
