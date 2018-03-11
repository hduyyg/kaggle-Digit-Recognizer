import logging
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, train_test_split
import functions


def get_train_model(data, label, path, flags):
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=0.1, random_state=42)
    best = {'score':-1,'n_components':-1,'C':-1,'gamma':-1}
    for n_components in range(20, 40):
        pca_model = PCA(n_components=n_components, whiten=True)
        pca_model.fit(x_train)
        new_x_train = pca_model.transform(x_train)
        new_x_test = pca_model.transform(x_test)
        logging.info('pca-n-components:{}'.format(n_components))

        for C in range(1, 5):
            for gamma in [1.0/new_x_train.shape[1], 0.1, 1]:
                svm_model = svm.SVC(C=C, gamma=gamma)
                svm_model.fit(new_x_train, y_train)
                score = svm_model.score(new_x_test, y_test)
                logging.info('svm:C={} gamma:{} score={}'.format(C, gamma, score))
                if score > best['score']:
                    best['score'] = score
                    best['n_components'] = n_components
                    best['C'] = 4
                    best['gamma'] = gamma
    logging.info(best)


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
