import jsonlines
import convenion
from elasticsearch import Elasticsearch
import json
import random
import glob
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn
import shutil

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from underthesea import word_tokenize


PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'
# PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QUESTION_ANSWER_INDEXER = './elastic/qa_indexer.jl'

URL_SEARCH = 'http://127.0.0.1:9200/qa_tgdd/_search'


def raw_index_file():
    with jsonlines.open(PATH_QUESTION_ANSWER_INDEXER, mode='w') as writer:
        with jsonlines.open(PATH_QUESTION_ANSWER) as reader:
            for qa in reader:
                if not convenion.is_valid_qa(qa):
                    continue
                id_doc = qa['id_cmt']
                question = qa['question']
                answer = qa['answer']
                question_custom = convenion.customize_string(question)
                answer_custom = convenion.customize_string(answer)
                question_removed_stopword = convenion.customize_and_remove_stopword(question)
                answer_removed_stopword = convenion.customize_and_remove_stopword(answer)
                # print(question_custom)
                # print(answer_custom)
                # print(question_removed_stopword)
                # print(answer_removed_stopword)
                doc_id = {"index": {"_id": id_doc}}
                doc = {
                    "question": question,
                    "answer": answer,
                    "question_custom": question_custom,
                    "answer_custom": answer_custom,
                    "question_removed_stopword": question_removed_stopword,
                    "answer_removed_stopword": answer_removed_stopword,
                }
                writer.write(doc_id)
                writer.write(doc)


def get_search_result(query_obj, page=0, size=10, field_search="question", **kwargs):
    es = Elasticsearch()
    body = {
        "query": {
            "match": {
                field_search: query_obj['question']
            }
        },
        # "from": 1,
        "size": 200
    }
    res = es.search(index='qa_tgdd', body=body)
    results = res['hits']
    current_hits = res['hits']['hits']
    raw_hits = []
    for hit in current_hits:
        if len(hit['_source']['question']) == 0 or len(hit['_source']['answer']) == 0:
            continue
        raw_hit = {
            "score": hit['_score'],
            "id": hit['_id'],
            "field_search": field_search,
            "question": hit['_source']['question'],
            "answer": hit['_source']['answer'],
            "relate_q_q": 0
        }
        print(raw_hit)
        raw_hits.append(raw_hit)

    raw_result = {
        "id_query": query_obj['id'],
        "total": results['total'],
        "total_current": len(results['hits']),
        "max_score": results['max_score'],
        "origin_question": query_obj['question'],
        "hits": raw_hits
    }

    return raw_result
    # with open('search_results_exp.json', 'w') as outfile:
    #     json.dump(raw_result, outfile)


def refresh_query_pool():
    # safe to run
    # paths1 = glob.glob('elastic/judged/2hardquestion/*.json')
    # paths2 = glob.glob('elastic/judged/ezquestion/*.json')
    # paths = glob.glob('elastic/judged/tmp/*/*.json') + paths1 + paths2

    paths = glob.glob('elastic/judged/tmp/*/*.json')

    arr_query = []
    for path in paths:
        with open(path, 'r') as f:
            print(path)
            search_result = json.load(f)

            query = {
                'id': search_result['id_query'],
                'question': search_result['origin_question'],
                'searched': 1
            }

            arr_query.append(query)

    with open('elastic/query-pool/query_pool.json', 'w+') as f:
        json.dump(arr_query, f)


def raw_query_pool():
    with open('elastic/query-pool/query_pool.json') as f:
        queries = json.load(f)
        print("Current queries len: ", len(queries))
        print("\n")
        arr_id = [query['id'] for query in queries]
        arr_id_checked = list(arr_id)

        arr_question_source = []
        with jsonlines.open(PATH_QUESTION_ANSWER) as reader:
            for qa in reader:
                if not convenion.is_valid_qa(qa):
                    continue
                arr_question_source.append(qa)
            print(random.choice(arr_question_source))

        user_judge = ''

        while (len(arr_id) != 250) and (user_judge != '0'):
            qa_checking = random.choice(arr_question_source)
            if qa_checking['id_cmt'] in arr_id_checked:
                continue
            arr_id_checked.append(qa_checking['id_cmt'])
            # print("Question: %(question)s\n" %qa_checking)
            # print('Input your jugde for quenstion: ')
            user_judge = input(qa_checking['question'] + '\n')
            if user_judge != '1':
                print("Collecting next question...\n")
                continue
            print("Add to query...\n")
            arr_id.append(qa_checking['id_cmt'])
            queries.append({
                'id': qa_checking['id_cmt'],
                'question': qa_checking['question'],
                'searched': 0
            })
            print("Current queries len: ", len(queries))
            print("\n")

        with open('elastic/query-pool/query_pool.json', 'w') as outfile:
            json.dump(queries, outfile)


def train_dev_test_split(X):
    train_dev, test = sklearn.model_selection.train_test_split(X, test_size=1 / 6, shuffle=True)
    train, dev = sklearn.model_selection.train_test_split(train_dev, test_size=1 / 5, shuffle=True)
    return train, dev, test


def split_data(path_glob):
    # Split judged results to 3 set: train, dev, test and save them to a specific dict
    # judged is separated with earch other and need to raw to one text file
    paths = glob.glob(path_glob)

    train_paths, dev_paths, test_paths = train_dev_test_split(paths)

    for path in train_paths:
        destination_dict = 'data/tmp/train'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)

    for path in dev_paths:
        destination_dict = 'data/tmp/dev'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)

    for path in test_paths:
        destination_dict = 'data/tmp/test'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)


def raw_pool(kind='train', pool='pool2', max_judged=30, strict=False):
    # Change kind variable to train, dev, test
    # kind: train, dex, test
    # pool: pool1 - pool10
    explicit_path_use = 'data/' + pool + '/' + kind + '/*.json'
    explicit_path_raw = 'data/' + pool + '/raw/' + kind + '.txt'

    raw_to_file(strict=strict,
                separator='\t',
                max_judged=max_judged,
                more_info=False,
                explicit_path_use=explicit_path_use,
                explicit_path_raw=explicit_path_raw)


def raw_to_file(strict=False, tokenize=False, separator='\t\t\t', max_judged=None, more_info=False,
                explicit_path_use=None, explicit_path_raw=None):
    # Raw all judged result in specific path to a text file
    # Modify PATH_USED
    # Raw data from json to csv like files
    PATH_TRAIN_REGEX = 'data/pool1/train/*.json'
    PATH_DEV_REGEX = 'data/pool1/dev/*.json'
    PATH_TEST_REGEX = 'data/pool1/test/*.json'

    PATH_USED = PATH_TRAIN_REGEX
    PATH_RAW = 'data/dump.txt'

    if PATH_USED == PATH_TRAIN_REGEX:
        PATH_RAW = 'data/pool1/raw/train.txt'
    elif PATH_USED == PATH_DEV_REGEX:
        PATH_RAW = 'data/pool1/raw/dev.txt'
    elif PATH_USED == PATH_TEST_REGEX:
        PATH_RAW = 'data/pool1/raw/test.txt'

    if more_info:
        if PATH_USED == PATH_TRAIN_REGEX:
            PATH_RAW = 'data/pool1/raw/train-moreinfo.txt'
        elif PATH_USED == PATH_DEV_REGEX:
            PATH_RAW = 'data/pool1/raw/dev-more-info.txt'
        elif PATH_USED == PATH_TEST_REGEX:
            PATH_RAW = 'data/pool1/raw/test-more-info.txt'

    if explicit_path_use is not None:
        PATH_USED = explicit_path_use
        PATH_RAW = explicit_path_raw

    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')

    path_judgeds = glob.glob(PATH_USED)
    with open(PATH_RAW, 'w+') as raw_file:
        for path_judged in path_judgeds:
            with open(path_judged, 'r') as file_judged:
                print(path_judged)
                judged_result = json.load(file_judged)

                origin_question = judged_result['origin_question']
                # origin_question = convenion.customize_string(origin_question)
                id_origin_q = judged_result['id_query']
                max_score = judged_result['max_score']
                current_judged = 0
                for hit in judged_result['hits']:
                    judged_question = hit['question']
                    # judged_question = convenion.customize_string(judged_question)
                    id_judged_q = hit['id']
                    score_search = hit['score']

                    elastic_similar = (max_score - score_search) / max_score

                    vec_org_q = doc2vec_model.infer_vector(
                        simple_preprocess(word_tokenize(origin_question, format='text')))
                    vec_related_q = doc2vec_model.infer_vector(
                        simple_preprocess(word_tokenize(judged_question, format='text')))
                    cosin_distance = doc2vec_model.wv.cosine_similarities(vec_org_q, np.array([vec_related_q]))[0]

                    label = '0'
                    if hit['relate_q_q'] == 0:
                        continue

                    current_judged += 1

                    if hit['relate_q_q'] == 2:
                        label = '1'
                    elif hit['relate_q_q'] == 1 and not strict:
                        label = '1'
                    # print(hit['question'])
                    # print(hit['relate_q_q'])
                    # print(label)
                    # test = origin_question + '\t' + judged_question + '\t' + label
                    # print(test)

                    # Use it if raw test file with more information
                    # raw_file.write(id_origin_q + '\t' + origin_question + '\t' + judged_question + '\t' + label +
                    #                '\t' + str(score_search) + '\n')

                    # Default test file

                    if max_judged is not None:
                        if current_judged > max_judged:
                            break

                    if not more_info:
                        raw_file.write(origin_question + separator + judged_question + separator + label + '\n')

                        # current_judged += 1
                        # if max_judged is not None:
                        #     if current_judged >= max_judged:
                        #         break
                    else:
                        # print('more info')
                        raw_file.write(origin_question + separator + judged_question + separator + label + separator
                                       + str(elastic_similar) + separator + str(cosin_distance) + '\n')
                        #
                        # current_judged += 1
                        # if max_judged is not None:
                        #     if current_judged >= max_judged:
                        #         break
                file_judged.close()
    raw_file.close()


def search_by_query_pool(path_query_pool='elastic/query_pool-old.json', path_raw_result='./elastic/search_result/'):
    with open(path_query_pool) as f:
        queries = json.load(f)
        for query_obj in queries:
            if query_obj['searched'] != 0:
                continue
            raw_result = get_search_result(query_obj)
            path = path_raw_result + str(query_obj['id']) + '.json'
            with open(path, 'w') as outfile:
                json.dump(raw_result, outfile)


def statistic_search_result():
    judged_results_path = glob.glob("./elastic/judged/train/*.json")
    count_questions = len(judged_results_path)
    total_pair = 0
    total_good = 0
    total_useful = 0
    notyet_judged = 0
    total_bad = 0

    total_pair_2 = 0
    total_good_2 = 0
    total_useful_2 = 0
    notyet_judged_2 = 0
    total_bad_2 = 0

    for path in judged_results_path:
        with open(path, 'r') as f:
            # print(path)
            judged_result = json.load(f)
            # print(len(judged_result['hits']))
            # total_pair += len(judged_result['hits'])
            notyet_judged += len([question for question in judged_result['hits'] if question['relate_q_q'] == 0])
            total_good += len([question for question in judged_result['hits'] if question['relate_q_q'] == 2])
            total_useful += len([question for question in judged_result['hits'] if question['relate_q_q'] == 1])
            total_bad += len([question for question in judged_result['hits'] if question['relate_q_q'] == -1])
            # if notyet_judged > 0:
            #     print(path)
            #     break
            half_hits = judged_result['hits'][:len(judged_result['hits'])//2]
            notyet_judged_2 += len([question for question in half_hits if question['relate_q_q'] == 0])
            total_good_2 += len([question for question in half_hits if question['relate_q_q'] == 2])
            total_useful_2 += len([question for question in half_hits if question['relate_q_q'] == 1])
            total_bad_2 += len([question for question in half_hits if question['relate_q_q'] == -1])

    # print('notyet_judged: ', notyet_judged)
    print('total_question: ', count_questions)
    total_pair = total_good + total_useful + total_bad
    print('total_pair: ', total_pair)
    print('total_good: %d - %f' % (total_good, (total_good * 100 / total_pair)))
    print('total_useful: %d - %f' % (total_useful, (total_useful * 100 / total_pair)))
    print('total_bad: %d - %f' % (total_bad, (total_bad * 100 / total_pair)))
    print('------------------------------')
    total_pair_2 = total_good_2 + total_useful_2 + total_bad_2
    print('total_pair_half: ', total_pair_2)
    print('total_good_half: %d - %f' % (total_good_2, (total_good_2 * 100 / total_pair_2)))
    print('total_useful_half: %d - %f' % (total_useful_2, (total_useful_2 * 100 / total_pair_2)))
    print('total_bad_half: %d - %f' % (total_bad_2, (total_bad_2 * 100 / total_pair_2)))


def caculate_AP(path, strict, dict_path):
    # path: path to file json
    # strict: True or False. If True, only question related masked 2 is considered relevan
    # dict_path: dictionary path. Used for rename file json. Use not very offend
    with open(path, 'r') as f:
        search_result = json.load(f)
        hits = search_result['hits']
        arr_denote_all = []
        arr_denote_top30 = []
        arr_denote_top10 = []
        num_related = 0
        for hit in hits:
            if hit['relate_q_q'] == 3 or hit['relate_q_q'] == 0:
                continue
            elif hit['relate_q_q'] == 2:
                num_related += 1
                arr_denote_all.append(1)

                if len(arr_denote_top30) < 30:
                    arr_denote_top30.append(1)

                if len(arr_denote_top10) < 10:
                    arr_denote_top10.append(1)

            elif hit['relate_q_q'] == 1 and not strict:
                num_related += 1
                arr_denote_all.append(1)

                if len(arr_denote_top30) < 30:
                    arr_denote_top30.append(1)

                if len(arr_denote_top10) < 10:
                    arr_denote_top10.append(1)

            else:
                arr_denote_all.append(0)

                if len(arr_denote_top30) < 30:
                    arr_denote_top30.append(0)

                if len(arr_denote_top10) < 10:
                    arr_denote_top10.append(0)
        # newname = '{}-{:.2f}-{:.2f}-{:.2f}-{:d}.json'.format(search_result['id_query'],
        #                                                     convenion.caculate_AP(arr_denote_all),
        #                                                     convenion.caculate_AP(arr_denote_top30),
        #                                                     convenion.caculate_AP(arr_denote_top10),
        #                                                     num_related)
        newname = '{}-{:.2f}-{:d}.json'.format(search_result['id_query'],
                                               convenion.caculate_AP(arr_denote_top10),
                                               num_related)
        print(newname)
        os.rename(path, dict_path + '/' + newname)
        return convenion.caculate_AP(arr_denote_top10)


def caculate_mAP(dict_path, strict=False):
    # dict_path: dictionary path to json files
    # strict: True or False. If True, only question related masked 2 is considered relevan
    paths = glob.glob(dict_path + "/*.json")
    paths.sort()
    arr_AP = []
    for path in paths:
        arr_AP.append(caculate_AP(path, strict, dict_path))
    nparr_AP = np.array(arr_AP)
    print('mAP: ', nparr_AP.mean())
    print('max AP: ', nparr_AP.max())
    print('min AP: ', nparr_AP.min())
    print('radian AP: ', np.median(nparr_AP))

    print('sort: ', np.sort(nparr_AP))


def tmp():
    raw_pool(kind='train', pool='tmp', max_judged=10, strict=True)
    raw_pool(kind='dev', pool='tmp', max_judged=10, strict=True)
    raw_pool(kind='test', pool='tmp', max_judged=10, strict=True)

# raw_query_pool()
# search_by_query_pool(path_query_pool='elastic/query-pool/query_pool.json', path_raw_result='elastic/search_result/')
# statistic_search_result()
caculate_mAP('elastic/judged/tmp/ư_new', strict=False)


# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=None, more_info=True,
#             explicit_path_use='elastic/judged/ezquestion/*.json',
#             explicit_path_raw='data/pool1/raw/ez-moreinfo-strict.json')

# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/tmp/train/*.json',
#             explicit_path_raw='data/tmp/raw/train.txt')
# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/tmp/dev/*.json',
#             explicit_path_raw='data/tmp/raw/dev.txt')
# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/tmp/test/*.json',
#             explicit_path_raw='data/tmp/raw/test.txt')

# split_data(path_glob='elastic/judged/tmp/*/*.json')

refresh_query_pool()


# raw_query_pool()


