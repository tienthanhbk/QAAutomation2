import jsonlines
from src import convenion
from elasticsearch import Elasticsearch
import json
import random
import glob
import os
import numpy as np
import sklearn
import shutil

from gensim.models.doc2vec import Doc2Vec

PATH_NEW_QUESTIONS_GLOB = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/general/*.jl'
PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'
# PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QUESTION_ANSWER_INDEXER = 'elastic/qa_indexer.jl'

URL_SEARCH = 'http://127.0.0.1:9200/qa_tgdd/_search'


def raw_index_file2():
    with jsonlines.open(PATH_QUESTION_ANSWER_INDEXER, mode='w') as writer:

        file_paths = glob.glob('/Users/tienthanh/Projects/ML/datapool/tgdd_qa/general/*.jl')

        for path in file_paths:
            with jsonlines.open(path) as reader:
                for qa in reader:
                    # if not convenion.is_valid_qa(qa):
                    #     continue
                    id_doc = qa['id']
                    question = qa['question']
                    answer = '_'
                    # print(question_custom)
                    # print(answer_custom)
                    # print(question_removed_stopword)
                    # print(answer_removed_stopword)
                    doc_id = {"index": {"_id": id_doc}}
                    doc = {
                        "question": question,
                        "answer": answer,
                    }
                    writer.write(doc_id)
                    writer.write(doc)


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


def refresh_query_pool(path_searched='elastic/judged/tmp/*/*.json', path_pool='elastic/query-pool/query_pool.json'):
    # safe to run
    # paths1 = glob.glob('elastic/judged/2hardquestion/*.json')
    # paths2 = glob.glob('elastic/judged/ezquestion/*.json')
    # paths = glob.glob('elastic/judged/tmp/*/*.json') + paths1 + paths2

    paths = glob.glob(path_searched)
    # paths = glob.glob('elastic/judged/test-data/tmp/*.json') + glob.glob('elastic/judged/test-data/2ez/*.json') + \
    #         glob.glob('elastic/judged/test-data/2hard/*.json')

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

    with open(path_pool, 'w+') as f:
        json.dump(arr_query, f)


def raw_query_pool2():
    with open('/Users/tienthanh/Projects/ML/QAAutomation/elastic/anew_data/query_pool/query_train.json') as f:
        queries = json.load(f)
        print("Current queries len: ", len(queries))
        print("\n")
        arr_id = [query['id'] for query in queries]
        arr_id_checked = list(arr_id)

        arr_question_source = []
        for path in glob.glob(PATH_NEW_QUESTIONS_GLOB):
            with jsonlines.open(path) as reader:
                for qa in reader:
                    # if not convenion.is_valid_qa(qa):
                    #     continue
                    arr_question_source.append(qa)

        # print(random.choice(arr_question_source))

        user_judge = ''

        while (len(arr_id) != 4320) and (user_judge != '0'):
            qa_checking = random.choice(arr_question_source)
            if qa_checking['id'] in arr_id_checked:
                continue
            arr_id_checked.append(qa_checking['id'])
            # print("Question: %(question)s\n" %qa_checking)
            # print('Input your jugde for quenstion: ')
            # user_judge = input(qa_checking['question'] + '\n')
            user_judge = '1'
            if user_judge != '1':
                print("Collecting next question...\n")
                continue
            print("Add to query...\n")
            arr_id.append(qa_checking['id'])
            queries.append({
                'id': qa_checking['id'],
                'question': qa_checking['question'],
                'searched': 0
            })
            print("Current queries len: ", len(queries))
            print("\n")

        with open('/Users/tienthanh/Projects/ML/QAAutomation/elastic/anew_data/query_pool/query_train.json', 'w') as outfile:
            json.dump(queries, outfile, indent=3, ensure_ascii=False)


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


def train_dev_test_split(X, test=True):
    if test:
        train_dev, test = sklearn.model_selection.train_test_split(X, test_size=1 / 8, shuffle=True)
        train, dev = sklearn.model_selection.train_test_split(train_dev, test_size=1 / 7, shuffle=True)
        return train, dev, test

    train, dev = sklearn.model_selection.train_test_split(X, test_size=1 / 5, shuffle=True)
    return train, dev


def split_data(path_glob, test=True):
    # Split judged results to 3 set: train, dev, test and save them to a specific dict
    # judged is separated with earch other and need to raw to one text file
    paths = glob.glob(path_glob)
    train_paths = []
    dev_paths = []
    test_paths = []
    if test:
        train_paths, dev_paths, test_paths = train_dev_test_split(paths, test=test)
    else:
        train_paths, dev_paths = train_dev_test_split(paths, test=test)

    for path in train_paths:
        destination_dict = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/train'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)

    for path in dev_paths:
        destination_dict = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/dev'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)

    for path in test_paths:
        destination_dict = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/test'
        filename = os.path.basename(path)
        shutil.copyfile(path, destination_dict + '/' + filename)


def raw_pool(kind='train', pool='similar2', max_judged=10, strict=False):
    # Change kind variable to train, dev, test
    # kind: train, dex, test
    # pool: similar1 - pool10
    # explicit_path_use = 'data/' + pool + '/' + kind + '/*.json'
    # explicit_path_raw = 'data/' + pool + '/raw/' + kind + '.txt'

    explicit_path_use = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/test' \
                        '/*.json'
    explicit_path_raw = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/raw/test.txt'

    raw_to_file(strict=strict,
                separator='\t',
                max_judged=max_judged,
                more_info=False,
                explicit_path_use=explicit_path_use,
                explicit_path_raw=explicit_path_raw,
                tokenize=False)


def raw_to_file(strict=False, tokenize=False, separator='\t\t\t', max_judged=None, more_info=False,
                explicit_path_use=None, explicit_path_raw=None):
    # Raw all judged result in specific path to a text file
    # Modify PATH_USED
    # Raw data from json to csv like files

    num_positive = 0
    num_negative = 0

    PATH_TRAIN_REGEX = 'data/similar1/train/*.json'
    PATH_DEV_REGEX = 'data/similar1/dev/*.json'
    PATH_TEST_REGEX = 'data/similar1/test/*.json'

    PATH_USED = PATH_TRAIN_REGEX
    PATH_RAW = 'data/dump.txt'

    if PATH_USED == PATH_TRAIN_REGEX:
        PATH_RAW = 'data/similar1/raw/train.txt'
    elif PATH_USED == PATH_DEV_REGEX:
        PATH_RAW = 'data/similar1/raw/dev.txt'
    elif PATH_USED == PATH_TEST_REGEX:
        PATH_RAW = 'data/similar1/raw/test.txt'

    if more_info:
        if PATH_USED == PATH_TRAIN_REGEX:
            PATH_RAW = 'data/similar1/raw/train-moreinfo.txt'
        elif PATH_USED == PATH_DEV_REGEX:
            PATH_RAW = 'data/similar1/raw/dev-more-info.txt'
        elif PATH_USED == PATH_TEST_REGEX:
            PATH_RAW = 'data/similar1/raw/test-more-info.txt'

    if explicit_path_use is not None:
        PATH_USED = explicit_path_use
        PATH_RAW = explicit_path_raw

    # doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')

    path_judgeds = glob.glob(PATH_USED)
    with open(PATH_RAW, 'w+') as raw_file:
        for path_judged in path_judgeds:
            with open(path_judged, 'r') as file_judged:
                print(path_judged)
                judged_result = json.load(file_judged)

                origin_question = judged_result['origin_question']

                if tokenize:
                    origin_question = convenion.customize_string(origin_question)
                else:
                    origin_question = convenion.simple_customize_string(origin_question)
                id_origin_q = judged_result['id_query']
                current_judged = 0
                for hit in judged_result['hits']:
                    current_judged += 1

                    judged_question = hit['question']
                    if tokenize:
                        judged_question = convenion.customize_string(judged_question)
                    else:
                        judged_question = convenion.simple_customize_string(judged_question)

                    # Check duplicate
                    if hit['id'] == judged_result['id_query']:
                        continue

                    score_search = hit['score']

                    label = '0'

                    if hit['relate_q_q'] == 2:
                        label = '1'
                    elif hit['relate_q_q'] == 1 and not strict:
                        label = '1'

                    if max_judged is not None:
                        if current_judged > max_judged:
                            break

                    if hit['relate_q_q'] == 2:
                        num_positive += 1
                    elif hit['relate_q_q'] == 1 and not strict:
                        num_positive += 1
                    else:
                        num_negative += 1

                    if not more_info:
                        raw_file.write(origin_question + separator + judged_question + separator + label + '\n')

                    else:
                        print('more info')
                        raw_file.write(origin_question + separator + judged_question + separator + label + separator
                                       + str(id_origin_q) + separator + str(score_search) + '\n')

                file_judged.close()
    print('num positive: ', num_positive)
    print('num_negative: ', num_negative)
    raw_file.close()


def get_search_result(query_obj, index, page=0, size=10, field_search="question", **kwargs):
    es = Elasticsearch()
    body = {
        "query": {
            "match": {
                field_search: query_obj['question']
            }
        },
        # "from": 1,
        "size": 100
    }
    res = es.search(index=index, body=body)
    results = res['hits']
    current_hits = res['hits']['hits']
    raw_hits = []
    for hit in current_hits:
        if len(hit['_source']['question']) == 0 or len(hit['_source']['answer']) == 0:
            continue
        # Ignore self
        if query_obj['id'] == hit['_id']:
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


def search_by_query_pool(path_query_pool='elastic/query_pool-old.json', path_raw_result='./elastic/search_result/',
                         index='qa_tgdd'):
    with open(path_query_pool) as f:
        queries = json.load(f)
        for query_obj in queries:
            if query_obj['searched'] != 0:
                continue
            query_obj['searched'] = 1
            raw_result = get_search_result(query_obj, index=index)
            path = path_raw_result + str(query_obj['id']) + '.json'
            with open(path, 'w') as outfile:
                json.dump(raw_result, outfile, indent=3, ensure_ascii=False)

        with open(path_query_pool, 'w') as outfile:
            json.dump(queries, outfile, indent=3, ensure_ascii=False)


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


def refresh_judged_files(judged_path, new_path):
    # dict_path: dictionary path to json files
    # strict: True or False. If True, only question related masked 2 is considered relevan
    paths = glob.glob(judged_path + "/*.json")
    paths.sort()

    for path in paths:
        with open(path, 'r') as f:
            search_result = json.load(f)
            hits = search_result['hits']
            for hit in hits:
                hit['relate_q_q'] = 0
            newname = '{}.json'.format(search_result['id_query'])
            print(newname)
            with open(new_path + '/' + newname, 'w+') as f:
                json.dump(search_result, f, indent=3, ensure_ascii=False)


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
            # if hit['relate_q_q'] == 3 or hit['relate_q_q'] == 0:
            #     continue
            try:
                if hit['relate_q_q'] == 2:
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
            except KeyError:
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


def caculate_positive_ratio(dict_path):
    paths = glob.glob(dict_path + "/*.json")
    num_positive = 0
    count_not_duplicated = 0
    for path in paths:
        with open(path, 'r') as f:
            search_result = json.load(f)
            hits = search_result['hits']
            arr_denote_all = []
            arr_denote_top30 = []
            arr_denote_top10 = []
            num_related = 0

            counter = 0

            for hit in hits:
                counter += 1
                if hit['id'] == search_result['id_query']:
                    continue
                # Check duplicate
                if counter <= 10:
                    count_not_duplicated = count_not_duplicated + 1
                try:
                    if hit['relate_q_q'] == 2:
                        num_related += 1
                        arr_denote_all.append(1)

                        if counter <= 30:
                            arr_denote_top30.append(1)

                        if counter <= 10:
                            num_positive += 1
                            arr_denote_top10.append(1)

                    elif hit['relate_q_q'] == 1:
                        num_related += 1
                        arr_denote_all.append(1)

                        if counter <= 30:
                            arr_denote_top30.append(1)

                        if counter <= 10:
                            num_positive += 1
                            arr_denote_top10.append(1)

                    else:
                        arr_denote_all.append(0)

                        if counter <= 30:
                            arr_denote_top30.append(0)

                        if counter <= 10:
                            arr_denote_top10.append(0)
                except KeyError:
                    count_not_duplicated += 1

                    arr_denote_all.append(0)

                    if counter <= 30:
                        arr_denote_top30.append(0)

                    if counter <= 10:
                        arr_denote_top10.append(0)
    print('(len(paths) * 10): ', (len(paths) * 10))
    print('count_not_duplicated_total: ', count_not_duplicated)
    return num_positive / count_not_duplicated



def tmp():
    raw_pool(kind='all', pool='newdata', max_judged=10, strict=False)
    # raw_pool(kind='dev', pool='tmp', max_judged=10, strict=True)
    # raw_pool(kind='test', pool='tmp', max_judged=10, strict=True)


def get_origin_q_with_positive_q():
    base_path = '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/persion/Tri_part1'
    paths = glob.glob(base_path + "/*.json")
    for path in paths:
        print(path)
        with open(path, 'r') as f:
            search_result = json.load(f)
            hits = search_result['hits']
            num_related = 0

            for hit in hits:
                try:
                    # Check duplicate
                    if hit['id'] == search_result['id_query']:
                        continue

                    if hit['relate_q_q'] == 2:
                        num_related += 1
                    elif hit['relate_q_q'] == 1:
                        num_related += 1
                except KeyError:
                    continue

            print(num_related)
            if num_related > 0:
                shutil.copy2(path, base_path + '/positiveable')
            else:
                shutil.copy2(path, base_path + '/allnegative')


def denote_consensus(path_judge, path_target_dict):
    judge = None
    target=None

    arr_denote = []
    with open(path_judge) as f:
        judge = json.load(f)

    for path_target in glob.glob(path_target_dict + '/*.json'):
        with open(path_target) as f:
            target = json.load(f)
            if target['id_query'] == judge['id_query']:
                break
            else:
                target = None

    if target is None:
        return arr_denote

    count = 0

    for judge_q in judge['hits']:
        count += 1
        for target_q in target['hits']:
            if judge_q['id'] == target_q['id']:
                if judge_q['relate_q_q'] == target_q['relate_q_q']: # True
                    if judge_q['relate_q_q'] == 1: # True positive
                        arr_denote.append(1)
                    else: # True negative
                        arr_denote.append(0)
                else: # False
                    arr_denote.append(-1)
                break
        if count >= 10:
            break

    return arr_denote


def caculate_consensus(arr_denote):
    if len(arr_denote) == 0:
        return None
    num_true = 0
    for denote in arr_denote:
        if denote == 1 or denote == 0:
            num_true += 1
    return num_true / len(arr_denote)

# def remove_duplicate_from_file(path, dest_dict):
#     # Remove duplicate question from path and cut to dest_dict
#     f = open(path)
#     jsonF = json.load(f)
#     id_origin = jsonF['id_query']


# arr = denote_consensus('/Users/tienthanh/Desktop/Tienthanh-Chi_Thanh/today/593993.json',
#                  '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/positiveable/all')
# print(caculate_consensus(arr))

def caculate_consensus_all(path_judge_dict, path_target_dict):
    arr_consensus = []
    for path_judge in glob.glob(path_judge_dict + '/*.json'):
        score_consensus = caculate_consensus(denote_consensus(path_judge, path_target_dict))
        if score_consensus is not None:
            arr_consensus.append(score_consensus)
    print("len: ", len(arr_consensus))
    return np.mean(arr_consensus)


# print(caculate_consensus_all('/Users/tienthanh/Desktop/Tienthanh-Chi_Thanh/today',
#                              '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/positiveable/*'))

# get_origin_q_with_positive_q()

# tmp()

# raw_index_file2()
# raw_query_pool2()
# search_by_query_pool(path_query_pool='elastic/query-pool/query_test.json',
#                      path_raw_result='elastic/search_result/test/')

# Search general data:gensim/model/question.d2v
# search_by_query_pool(path_query_pool='/Users/tienthanh/Projects/ML/QAAutomation/elastic/anew_data/query_pool/query_train.json',
#                      path_raw_result='/Users/tienthanh/Projects/ML/QAAutomation/elastic/anew_data/search_result/search/',
#                      index='qa_general')

# statistic_search_result()
# caculate_mAP('/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/positiveable/split/test', strict=False)


# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=None, more_info=True,
#             explicit_path_use='elastic/judged/ezquestion/*.json',
#             explicit_path_raw='data/similar1/raw/ez-moreinfo-strict.json')
# raw_to_file(strict=False, tokenize=False, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/pool1/dev/*.json',
#             explicit_path_raw='data/pool1/raw/dev.txt')
# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/pool5/dev/*.json',
#             explicit_path_raw='data/pool5/raw/dev.txt')
#
# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/pool4/train/*.json',
#             explicit_path_raw='data/pool4/raw/train.txt')
# raw_to_file(strict=False, tokenize=True, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='data/pool4/dev/*.json',
#             explicit_path_raw='data/pool4/raw/dev.txt')
# raw_to_file(strict=False, tokenize=False, separator='\t', max_judged=10, more_info=False,
#             explicit_path_use='elastic/judged/test-data/tmp/*.json',
#             explicit_path_raw='data/test_data/raw/test.txt')

# split_data(path_glob='/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/all/*.json', test=True)
# print(caculate_positive_ratio('/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/train'))
# print(caculate_positive_ratio('/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/dev'))
print(caculate_positive_ratio('/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/all/positiveable/split/test'))
# refresh_query_pool()
# refresh_query_pool(path_searched='elastic/anew_data/search_result/train/*.json',
#                    path_pool='elastic/anew_data/query_pool/query_train.json')


# raw_query_pool()

# for path in glob.glob('/Users/tienthanh/Projects/ML/QAAutomation/data/8/*.json'):
#     print(path)
#     f = open(path, 'r')
#     search_result = json.load(f)
#     f_new = open(path, 'w')
#     json.dump(search_result, f_new, indent=3, ensure_ascii=False)

# refresh_judged_files('/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/judged/Tri_part1',
#                      '/Users/tienthanh/Projects/ML/QAAutomation/data/newdata/origin/Tri_origin')