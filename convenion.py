import os.path
import json_lines
import numpy as np
import re
from underthesea import word_tokenize

# PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'
PATH_TO_STOPWORDS = '/Users/tienthanh/Projects/ML/QAAutomation/vietnamese-stopwords-dash.txt'


def get_qa_by_id(arr_id_cmt):
    with open(PATH_QUESTION_ANSWER) as f:
        for qa in json_lines.reader(f):
            if qa['id_cmt'] in arr_id_cmt:
                yield qa


def union_multi_arr(*args):
    return set().union(*args)


def save_result_to_file(query, results_id):
    if not os.path.exists('search_result'):
        os.mkdir('search_result')


def is_valid_qa(qa):
    if (qa['question'] is None) or (qa['answer'] is None) or (qa['id_cmt'] is None or (len(qa['question']) == 0) or (len(qa['answer']) == 0)):
        return False
    return True


def customize_string(string):
    string = string.lower()
    string = re.sub(r'\bcmt\b', 'chứng minh thư', string)
    string = re.sub(r'\bshk\b', 'sổ hộ khẩu', string)
    string = re.sub(r'\bđt\b', 'điện thoại', string)
    string = re.sub(r'\bdt\b', 'điện thoại', string)
    string = re.sub(r'\bdc\b', 'được', string)
    string = re.sub(r'\bdk\b', 'được', string)
    string = re.sub(r'\bđk\b', 'được', string)
    string = re.sub(r'\bđc\b', 'được', string)
    string = re.sub(r'\bnhiu\b', 'nhiêu', string)
    string = re.sub(r'\bbn\b', 'bao nhiêu', string)
    string = re.sub(r'\bbnhieu\b', 'bao nhiêu', string)
    string = re.sub(r'\bk\b', ' không', string)
    string = re.sub(r'\bsp\b', 'sản phẩm', string)
    string = re.sub(r'\blác\b', 'lag', string)
    string = re.sub(r'\b0d\b', 'không đồng', string)
    string = re.sub(r'\b0đ\b', 'không đồng', string)
    string = re.sub(r'\b0 d\b', 'không đồng', string)
    string = re.sub(r'\b0 đ\b', 'không đồng', string)
    string = re.sub(r'\b12\b', ' mười_hai ', string)
    string = re.sub(r'\b10\b', ' mười', string)
    string = re.sub(r'\b9\b', ' chín', string)
    string = re.sub(r'\bngắc\b', 'ngắt', string)
    string = re.sub(r'\bsetting\b', 'cấu hình', string)
    string = re.sub(r'\bmax\b', 'cao nhất', string)
    string = re.sub(r'\bbóc hộp\b', 'mói', string)
    string = re.sub(r'\bmở hộp\b', 'mới', string)
    string = re.sub(r'\bhđh\b', 'hệ điều hành', string)
    string = re.sub(r'\biphon\b', 'iphone', string)
    string = re.sub(r'\bip\b', 'iphone', string)
    string = re.sub(r'\bios11\b', 'ios mười_một', string)
    string = re.sub(r'\bios10\b', 'ios mười', string)
    string = re.sub(r'\bios9\b', 'ios chín', string)
    string = re.sub(r'\bios12\b', 'ios mười_hai', string)
    string = re.sub(r'\b10%\b', 'mười phần_trăm', string)
    string = re.sub(r'\b15%\b', 'mười_năm phần_trăm', string)
    string = re.sub(r'\b20%\b', 'hai_mươi phần_trăm', string)
    string = re.sub(r'\b25%\b', 'hai_năm phần_trăm', string)
    string = re.sub(r'\b30%\b', 'ba_mươi phần_trăm', string)
    string = re.sub(r'\b35%\b', 'ba_năm phần_trăm', string)
    string = re.sub(r'\b40%\b', 'bốn_mươi phần_trăm', string)
    string = re.sub(r'\b50%\b', 'năm_mươi phần_trăm', string)
    string = re.sub(r'\b60%\b', 'sáu_mưoi phần_trăm', string)
    string = re.sub(r'\b20\b', 'hai_mươi', string)
    string = re.sub(r'\b30\b', 'ba_mươi', string)
    string = re.sub(r'\b40\b', 'bốn_mươi', string)
    string = re.sub(r'\b50\b', 'năm_mươi', string)
    string = re.sub(r'\b60\b', 'sáu_mươi', string)
    string = re.sub(r'\b5\b', 'năm', string)
    string = re.sub(r'\b0d\b', 'không trả trước', string)
    string = re.sub(r'\b0%\b', 'không lãi suất', string)
    string = re.sub(r'\b0 %\b', 'không lãi suất', string)
    string = re.sub(r'\b0đ\b', 'không trả trước', string)
    string = re.sub(r'\b0\b', 'không', string)
    string = re.sub(r'%', ' phần_trăm ', string)

    string = string.replace('\xa0', ' ')\
        .replace('.', ' ').replace(',', ' ')\
        .replace('?', ' ').replace('!', ' ')\
        .replace('/', ' ').replace('-', '_') \
        .replace(':', ' ') \
        .strip()
    string = re.sub('\s+', ' ', string).strip()
    return word_tokenize(string, format="text")


def get_stopwords():
    with open(PATH_TO_STOPWORDS, 'r') as f:
        return f.read().splitlines()


def remove_stopword(string):
    arr_stopword = get_stopwords()
    arr_str = string.split(' ')
    for str in arr_str:
        if str in arr_stopword:
            string = string.replace(str, '')
    string = re.sub('\s+', ' ', string).strip()
    return string


def customize_and_remove_stopword(string):
    string = customize_string(string)
    string = remove_stopword(string)
    return string


def caculate_AP(arr):
    # P_at_k: P@k
    P_at_k = 0
    relevan_len = len([val for val in arr if val == 1])
    precision_threshold = 0
    for index, val in enumerate(arr):
        if val == 1:
            precision_threshold += 1
            P_at_k += precision_threshold / (index + 1)
    # AP = P_at_k / len(relevan_doc)
    if relevan_len == 0:
        return 0
    return P_at_k / relevan_len

# t = word_tokenize('xin chào đồng bào cả nước', format='text')
# t = caculate_AP([0,1,0, 0, 0, 1, 1, 1, 0, 0])
print(customize_string('Hỏ trợ trả góp bao nhiêu    1% lãi vậy ạ'))
