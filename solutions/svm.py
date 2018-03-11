import logging
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, train_test_split
import functions


def train_for_model(flags):   
    data = np.load('data/' + flags['train_data'])
    label = np.load('data/' + flags['train_label'])

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
    logging.info('best paras:{}'.format(best))

    pca_model = PCA(n_components=best['n_components'], whiten=True)
    pca_model.fit(data)
    data = pca_model.transform(data)
    np.save('data/train_data_resized_pca.npy', data)

    test_data = np.load('data/test_data_resized.npy')
    test_data = pca_model.transform(test_data)
    test_data = np.save('data/test_data_resized_pca.npy', test_data)
    
    model = svm.SVC(C=4)
    model.fit(data, label)
    joblib.dump(model, 'data/svm_pca.m')


def predict(flags):
    model_path = 'data/' + flags['train_model']
    model = joblib.load(model_path)    
    data_path = 'data/' + flags['test_data']
    data = np.load(data_path)
    label = model.predict(data)
    functions.save_result(label, 'xxxx')


def main(flags):
    if flags['command'] == 'train':
        train_for_model(flags)
    elif flags['command'] == 'predict':
        predict(flags)
    else:
        logging.error('illegal command!')


# command
# train
# predict