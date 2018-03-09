import logging
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import functions


def get_best_train_model(data, label, flags):
    best, path = 0, 'data/train_data_resized_knn_default.m'
    for n_neighbors in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(model, data, np.ravel(label), 
            cv=10, scoring='accuracy')
        score = scores.mean()
        logging.info('knn:n_neighbors={} scores={} \nscore={}'.format(
            n_neighbors, scores, score))
        if score > best:
            joblib.dump(model, path)
            best = score

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        scores = cross_val_score(model, data, np.ravel(label), 
            cv=10, scoring='accuracy')
        score = scores.mean()
        logging.info('knn:n_neighbors={} scores={} \nscore={}'.format(
            n_neighbors, scores, score))
        if score > best:
            joblib.dump(model, path)
            best = score
    model = joblib.load(path)
    model.fit(data, label)
    joblib.dump(model, path)


def get_train_model(data, label, path_prefix, flags):
    if flags['command'] != 'get_best_train_model':
        logging.info('get the default knn model.')
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(data, np.ravel(label))
        path = path_prefix + '_knn_default.m'
        joblib.dump(model, path)
        logging.info('save model to {}'.format(path))
        return

    if flags['command'] == 'get_best_train_model':
        get_best_train_model(data, label, flags)


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
        result_name = flags['train_model'] + '_result'
        test_data = np.load('data/' + flags['test_data'] + '.npy')
        predict(test_data, model, result_name, flags)

# command:
# get_best_train_model