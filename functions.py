import pandas as pd
import numpy as np
import logging
import random
from collections import Counter


def save_result(data, file_name):
    npy_path = 'data/{}.npy'.format(file_name)
    logging.info('start save result into {}.'.format(npy_path))
    np.save(npy_path, data)

    csv_path = 'data/{}.csv'.format(file_name)
    logging.info('start save result into {}.'.format(csv_path))
    info = {}
    info['ImageId'] = [i for i in range(1, len(data) + 1)]
    info['Label'] = data
    data_frame = pd.DataFrame(info)
    data_frame.to_csv(csv_path, index=False, sep=',')


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
