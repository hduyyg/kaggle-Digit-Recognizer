import logging
import get_args
import process_data
from solutions import *


# def main():
#     flags = process_args.get_flags()
#     data = functions.get_data(flags)
#     res = [None for _ in range(len(flags['solutions']))]
#     for i in range(len(flags['solutions'])):
#         res[i] = eval('%s.main(data)' % flags['solutions'][i])
#     ans = functions.cross_validation(res)
#     functions.save_result(ans, flags['result_file'])
    
#     if flags['test_scale'] is not None:
#     	logging.info('get the error rate!!!')
#         test_count = data['test_label'].shape[0]
#         for i in range(len(flags['solutions'])):
#             error_count = 0
#             for j in range(test_count):
#                 if res[i][j] != data['test_label'][j]: error_count += 1
#             logging.info('The error rate in %s is %f' 
#                 % (flags['solutions'][i], error_count/test_count))
#         error_count = 0
#         for j in range(test_count):
#             if ans[j] != data['test_label'][j]: error_count += 1
#         logging.info('The error rate in %s is %f' 
#             % ('ans', error_count/test_count))

def main():
    args = get_args.main()
    eval('{}.main(args)'.format(args['py']))


if __name__ == '__main__':
    main()
