import logging
import pandas
import numpy
import sklearn
from sklearn import neighbors, cross_validation

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename='digit_recognize_knn.log',
        filemode='w+')


def main():
    infos = pandas.read_csv('../data/train.csv')
    datas = infos.values[0:,1:]
    labels = infos.values[0:,0]
    
    for n_neighbors in range(6, 11):
        for cv in [5, 10, 20]:
            logging.info('start n_neighbors={} cv={}'.format(n_neighbors, cv))
            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            scores = cross_validation.cross_val_score(clf, datas, labels, cv=cv, scoring='accuracy')
            logging.info('score={}'.format(scores.mean()))


if __name__ == '__main__':
    main()
