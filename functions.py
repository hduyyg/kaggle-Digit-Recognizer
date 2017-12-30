import pandas
import numpy
import logging
import random
from collections import Counter


def get_data(flags):
    if flags['test_scale'] is None:
        data_tain = pandas.read_csv(flags['train_file'])
        data_test = pandas.read_csv(flags['test_file'])
        data = {}
        data['train_label'] = data_tain.values[0:, 0]
        data['train_data'] = data_tain.values[0:, 1:]
        data['test_data'] = data_test.values[0:, 0:]
        data['test_label'] = None
        logging.info('train_data.shape = %s' % str(data['train_data'].shape))
        logging.info('train_label.shape = %s' % str(data['train_label'].shape))
        logging.info('test_data.shape = %s' % str(data['test_data'].shape))
        return data

    data_tain = pandas.read_csv(flags['train_file'])
    train_label = data_tain.values[0:, 0]
    train_data = data_tain.values[0:, 1:]
    row_count = train_data.shape[0]
    logging.info('train_data.shape = %s' % str(train_data.shape))
    logging.info('train_label.shape = %s' % str(train_label.shape))

    data = {}
    test_case_count = int(row_count * flags['test_scale'][1] / 100)
    logging.info('test_case_count = %d' % test_case_count)
    data['test_data'] = [None for _ in range(test_case_count)]
    data['test_label'] = [None for _ in range(test_case_count)]
    test_index = random.sample(range(row_count), test_case_count)
    for i in range(test_case_count):
        data['test_data'][i] = train_data[test_index[i]]
        data['test_label'][i] = train_label[test_index[i]]

    train_case_count = int(row_count * flags['test_scale'][0] / 100)
    logging.info('train_case_count = %d' % train_case_count)
    data['train_data'] = [None for _ in range(train_case_count)]
    data['train_label'] = [None for _ in range(train_case_count)]
    remain_index = [None for _ in range(row_count - test_case_count)]
    remain_index_sum = 0
    for i in range(row_count):
        if i not in test_index:
            remain_index[remain_index_sum] = i
            remain_index_sum += 1
    train_index = random.sample(remain_index, train_case_count)
    for i in range(train_case_count):
        data['train_data'][i] = train_data[train_index[i]]
        data['train_label'][i] = train_label[train_index[i]]

    for key, value in data.items():
        data[key] = numpy.array(value)
    logging.info('train_data.shape = %s' % str(data['train_data'].shape))
    logging.info('train_label.shape = %s' % str(data['train_label'].shape))
    logging.info('test_data.shape = %s' % str(data['test_data'].shape))
    logging.info('test_label.shape = %s' % str(data['test_label'].shape))
    return data


def save_result(data, file_name):
    logging.info('start save_result(data, file_name)!!!!')
    info = {}
    info['ImageId'] = [ i for i in range(1, len(data) + 1)]
    info['Label'] = data
    data_frame = pandas.DataFrame(info)
    data_frame.to_csv(file_name, index=False, sep=',')

def cross_validation(matrix):
    logging.info('start cross_validation(matrix)!!!!')
    row_count = len(matrix)
    col_count = len(matrix[0])
    logging.info('row_count=%d col_count=%d' % (row_count, col_count))
    res = [None for _ in range(col_count)]
    tmp = [None for _ in range(row_count)]
    for col in range(col_count):
        for row in range(row_count):
            tmp[row] = matrix[row][col]
        res[col] = Counter(tmp).most_common()[0][0]
    return res
