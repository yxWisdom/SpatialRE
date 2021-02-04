import sys

from common_utils import mark_predict_error, to_srl_format

if __name__ == '__main__':
    # print(sys.argv)
    func = sys.argv[1]
    if func == '-ner':
        path = sys.argv[2]
        mark_predict_error(path)
    if func == "-srl":
        path = sys.argv[2]
        to_srl_format(path)
    if func == "-p_m":
        path = sys.argv[2]

    # if func == "-"
