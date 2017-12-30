import argparse
import logging
import json


def get_args():
    parser = argparse.ArgumentParser(
        description='Model training and inference')
    
    parser.add_argument(
        '-log_level',
        default='DEBUG',
        help='input the log level.')
    parser.add_argument(
        '-solutions',
        default=None,
        nargs='*',
        required=True,
        help='input the solution will be used.')
    parser.add_argument(
        '-test_scale',
        default=None,
        nargs=2,
        type=int,
        help=('input two integer x and y to determine the test data scale.'
            'x% of the train-data used to train and y% of the train-data used '
            'to test.So,the sum of x plus y should less than 100! If you do '
            'not add this parameter,I will think you want use all train-data'
            'to train,and get the result to submit.'))
    parser.add_argument(
        '-train_file',
        default='train.csv',
        help='input the name of train file.')
    parser.add_argument(
        '-test_file',
        default='test.csv',
        help='input the name of test file.')
    parser.add_argument(
        '-result_file',
        default='result.csv',
        help='input the name of result file.use "w+" to open file.')
    args = parser.parse_args()
    return args

def verify_args_test_scale(args):
    if args.test_scale is None: return
    if (args.test_scale[0] <= 0) or (args.test_scale[1] <= 0):
        raise ValueError('The test_scale should greater than 0.')
    if args.test_scale[0] + args.test_scale[1] > 100:
        raise ValueError('The sum of test_scale should less than 100.')


def get_flags():
    args = get_args()
    verify_args_test_scale(args)
    log_level = args.log_level.upper()
    logging.basicConfig(level=eval('logging.%s' % log_level),
        format='%(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

    labels = ['solutions', 'test_scale', 'train_file', 
        'test_file', 'result_file']
    flags = {}
    for label in labels: flags[label] = eval('args.%s' % label)
    flags['result_file'] = 'data/%s' % flags['result_file']
    flags['train_file'] = 'data/%s' % flags['train_file']
    flags['test_file'] = 'data/%s' % flags['test_file']

    logging.info('****************flags********************')
    flags_infos = json.dumps(flags, indent=2, default=str)
    logging.info(flags_infos)
    logging.info('****************flags********************')
    return flags
