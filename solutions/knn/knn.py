import logging
import numpy
import operator
from collections import Counter

def classify(test_case, data_train, labels, k):
    data_train_row_count = data_train.shape[0]
    difference = numpy.tile(test_case, (data_train_row_count, 1)) - data_train
    distance = (difference**2).sum(axis=1)**0.5
    sort_index = distance.argsort()

    if distance[sort_index[0]] == 0:
        return labels[sort_index[0]]

    nearest = [None for _ in range(k)]
    for i in range(k):
        nearest[i] = labels[sort_index[i]]
    res = Counter(nearest).most_common()[0][0]
    return res


def main(data):
    logging.info("start knn!!")
    labels = data['train_label']
    data_train = data['train_data']
    data_test = data['test_data']
    k = 1
    logging.info('labels.shape = %s' % str(labels.shape))
    logging.info('data_train.shape = %s' % str(data_train.shape))
    logging.info('data_test.shape = %s' % str(data_test.shape))
    logging.info('k = %d' % k)

    test_case_count = data_test.shape[0]
    res = [None for _ in range(test_case_count)]
    for i in range(test_case_count):
        res[i] = classify(data_test[i], data_train, labels, k)
        logging.debug("the %d-th classify_result is: %s " % (i, res[i]))
    return res

if __name__ == '__main__':
    main()