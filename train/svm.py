import logging
import pandas
import numpy
from sklearn.svm import SVC
from sklearn import cross_validation

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename='digit_recognize_svm.log',
        filemode='w+')


def main():
    infos = pandas.read_csv('../data/train.csv')
    datas = infos.values[0:,1:]
    labels = infos.values[0:,0]
    
    for c in range(1, 5):
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            logging.info('start c={} kernel={}'.format(c, kernel))
            clf = SVC(C=c, kernel=kernel)
            scores = cross_validation.cross_val_score(clf, datas, labels, cv=10, scoring='accuracy')
            logging.info('score={}'.format(scores.mean()))


if __name__ == '__main__':
    main()
