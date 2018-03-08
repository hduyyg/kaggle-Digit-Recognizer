import logging
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import functions


def get_train_model(data, label, path_prefix, flags):
    if flags['command'] != 'get_best_train_model':
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(data, np.ravel(label))
        joblib.dump(model, path_prefix + '_knn_default.m')
        return

    if flags['command'] == 'get_best_train_model':
        # TODO:Implement the function-get_best_train_model in knn
        pass


def predict(data, model, result_name, flags):
    label = model.predict(data)
    functions.save_result(label, result_name)


def main(flags):
    if flags['train_data'] is not None:
        logging.info('get the train model by {}'.format(flags['train_data']))
        path_prefix = 'data/' + flags['train_data']
        train_data = np.load(path_prefix + '.npy')
        train_label = np.load('data/train_label.npy')
        get_train_model(train_data, train_label, path_prefix, flags)
    
    if flags['test_data'] is not None:
        if flags['train_model'] is None:
            logging.error('please add the train_model argument to predict.')
            return
        model = joblib.load('data/' + flags['train_model'])
        result_name = 'data/' + flags['train_model'] + '_result'
        test_data = np.load('data/' + flags['test_data'] + '.npy')
        predict(test_data, model, result_name, flags)

# command:
# get_best_train_model