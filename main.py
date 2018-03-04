import logging
import get_args
import pre_process
from solutions import knn


def main():
    args = get_args.main()
    knn.main(args)
    # eval('{}.main(args)'.format(args['py']))


if __name__ == '__main__':
    main()
