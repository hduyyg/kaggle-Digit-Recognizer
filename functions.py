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
