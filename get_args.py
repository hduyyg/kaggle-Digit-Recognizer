import argparse
import logging
import json


def get_args():
    parser = argparse.ArgumentParser(
        description='Process arguments.')
    parser.add_argument(
        '-log_level',
        default='INFO',
        help='the log level.'
    )
    parser.add_argument(
        '-log_to_file',
        default='False',
        choices=('True', 'False'),
        help='determine whether or not to save log into file.'
    )
    parser.add_argument(
        '-log_file',
        default='default.log',
        help='the name of file to save log.'
    )
    parser.add_argument(
        '-py',
        required=True,
        help='the excute python file name without suffix \'.py\'.'
    )
    parser.add_argument(
        '-command',
        default='predict',
        choices=('train', 'predict'),
        help='determine to train the model or predict the ans for solutions.'
    )
    args = parser.parse_args()
    return args


def logging_config(args):
    log_level = eval('logging.{}'.format(args.log_level.upper()))
    save_to_file = eval(args.log_to_file)
    if save_to_file:
        logging.basicConfig(level=log_level,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename=args.log_file,
            filemode='w+'
        )
    else:
        logging.basicConfig(level=log_level,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')


def main():
    args = get_args()
    logging_config(args)
    args = args.__dict__
    return args


if __name__ == '__main__':
    args = main()
    logging.info(json.dumps(args, indent=1, default=str))
