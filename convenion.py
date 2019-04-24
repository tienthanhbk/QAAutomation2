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
    replacer_arr = ['.', ',', '?', '\xa0', '\t']
    string = string.lower().replace('\xa0', ' ')\
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
