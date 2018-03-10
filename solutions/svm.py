import logging
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV
import functions


def get_best_train_model(data, label, flags):
    pass


def get_train_model(data, label, path, flags):
    model = svm.SVC(C=20, gamma=0.5)
    scores = cross_val_score(model, data, np.ravel(label), 
        cv=10, scoring='accuracy')
    score = scores.mean()
    logging.info('svm:scores={} \nscore={}'.format(scores, score))
    model.fit(data, label)
    joblib.dump(model, path)
    # best = 0
    # for C in range(1, 100):
    #     model = svm.SVC(C=C)
    #     scores = cross_val_score(model, data, np.ravel(label), 
    #         cv=10, scoring='accuracy')
    #     score = scores.mean()
    #     if score > best:
    #         best = score
    #         joblib.dump(model, path)
    #     logging.info('svm:C={} scores={} \nscore={}'.format(
    #         C, scores, score))

    #     for gamma in [0.01, 0.1, 1, 10]:
    #         model = svm.SVC(C=C, gamma=gamma)
    #         scores = cross_val_score(model, data, np.ravel(label), 
    #             cv=10, scoring='accuracy')
    #         score = scores.mean()
    #         if score > best:
    #             best = score
    #             joblib.dump(model, path)
    #         logging.info('svm:C={} scores={} \nscore={} gamma={}'.format(
    #             C, scores, score, gamma))
    # model = joblib.load(path)
    # model.fit(data, label)
    # joblib.dump(model, path)



def predict(data, model, result_name, flags):
    label = model.predict(data)
    functions.save_result(label, result_name)


def main(flags):
    if flags['train_data'] is not None:
        logging.info('get the train model by {}'.format(flags['train_data']))
        path_prefix = 'data/' + flags['train_data']
        train_data = np.load(path_prefix + '.npy')
        train_label = np.load('data/train_label.npy')
        model_path = path_prefix + '_svm.m'
        get_train_model(train_data, train_label, model_path, flags)
    
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