import logging
import get_args
import pre_process
from solutions import knn, svm, deep


def main():
    args = get_args.main()
    eval('{}.main(args)'.format(args['py']))


if __name__ == '__main__':
    main()
