import numpy as np
import pandas as pd
import convenion
import json
import glob
import os.path
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from underthesea import word_tokenize


def train_model(X_train, y_train, validation_data):
    # Initialize the constructor
    model = Sequential()

    # Add an input layer
    model.add(Dense(200, activation='relu', input_shape=(400,)))

    # Add one hidden layer
    model.add(Dense(100, activation='relu'))

    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    print('Training classify model..')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=validation_data, epochs=20, batch_size=1, verbose=1)

    model.save('model/simple_classify_model.h5')  # creates a HDF5 file 'my_model.h5'


def test_model():
    X_test = np.load('model/X_test.npy')
    y_test = np.load('model/y_test.npy')
    model = load_model('model/simple_classify_model.h5')

    y_pred = model.predict(X_test)
    print(y_pred[:10])
    score = model.evaluate(X_test, y_test, verbose=1)
    print(score)

#
# def raw_test_result():
#
#
# def caculate_MAP():




def raw_data(path_regex):
    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')
    PATH_JUDGED_EXAMPLE = 'elastic/judged/train/17630743-0.43-0.86-0.98.json'

    X_raw = []
    y_raw = []

    path_judgeds = glob.glob(path_regex)

    for path_judged in path_judgeds:
        with open(path_judged, 'r') as f:
            judged_result = json.load(f)

            origin_question = judged_result['origin_question']
            org_q_vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(origin_question, format='text')))
            org_q_vector = np.array(org_q_vector)

            for hit in judged_result['hits']:
                judged_question = hit['question']

                judged_q_vector = doc2vec_model.infer_vector(
                    simple_preprocess(word_tokenize(judged_question, format='text')))
                judged_q_vector = np.array(judged_q_vector)

                concat_vector = np.concatenate((org_q_vector, judged_q_vector))

                label = 0
                if hit['relate_q_q'] == 1 or hit['relate_q_q'] == 2:
                    label = 1

                X_raw.append(concat_vector)
                y_raw.append(label)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    return X_raw, y_raw


def save_raw_data():
    X_train, y_train = raw_data('./elastic/judged/train/*.json')
    X_dev, y_dev = raw_data('./elastic/judged/dev/*.json')
    X_test, y_test = raw_data('./elastic/judged/test/*.json')

    print('X_train len: ', len(X_train))
    print('y_train len: ', len(y_train))

    print('X_dev len: ', len(X_dev))
    print('y_dev len: ', len(y_dev))

    print('X_test len: ', len(X_test))
    print('y_test len: ', len(y_test))

    np.save('model/X_train.npy', X_train)
    np.save('model/X_dev.npy', X_dev)
    np.save('model/X_test.npy', X_test)

    np.save('model/y_train.npy', y_train)
    np.save('model/y_dev.npy', y_dev)
    np.save('model/y_test.npy', y_test)


def load_raw_data():
    X_train = np.load('model/X_train.npy')
    X_dev = np.load('model/X_dev.npy')
    X_test = np.load('model/X_test.npy')

    y_train = np.load('model/y_train.npy')
    y_dev = np.load('model/y_dev.npy')
    y_test = np.load('model/y_test.npy')

    print('X_train len: ', len(X_train))
    print('y_train len: ', len(y_train))

    print('X_dev len: ', len(X_dev))
    print('y_dev len: ', len(y_dev))

    print('X_test len: ', len(X_test))
    print('y_test len: ', len(y_test))

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def create_validation_data(X_dev, y_dev):
    validation_data = []
    for sample, label in zip(X_dev, y_dev):
        validation_data.append((sample, label))
    return validation_data

# Split the data up in train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
#
# train_model(X_train, y_train, X_test, y_test)


# X_train, X_dev, X_test, y_train, y_dev, y_test = load_raw_data()
#
# validation_data = create_validation_data(X_dev, y_dev)
#
# train_model(X_train, y_train, (X_dev, y_dev))
# test_model()
# load_raw_data()

<<<<<<< HEAD
=======
def raw_to_file(strict=False, tokenize=False, separator='\t\t\t'):
    # Raw data from json to csv like files
    PATH_TRAIN_REGEX = './elastic/judged/train/*.json'
    PATH_DEV_REGEX = './elastic/judged/dev/*.json'
    PATH_TEST_REGEX = './elastic/judged/test/*.json'

    PATH_USED = PATH_TRAIN_REGEX
    PATH_RAW = 'data/dump.txt'

    if PATH_USED == PATH_TRAIN_REGEX:
        PATH_RAW = 'data/train.txt'
    elif PATH_USED == PATH_DEV_REGEX:
        PATH_RAW = 'data/dev.txt'
    elif PATH_USED == PATH_TEST_REGEX:
        PATH_RAW = 'data/test.txt'

    path_judgeds = glob.glob(PATH_USED)
    with open(PATH_RAW, 'w+') as raw_file:
        for path_judged in path_judgeds:
            with open(path_judged, 'r') as file_judged:
                judged_result = json.load(file_judged)

                origin_question = judged_result['origin_question']
                origin_question = convenion.customize_string(origin_question)
                id_origin_q = judged_result['id_query']

                for hit in judged_result['hits']:
                    judged_question = hit['question']
                    judged_question = convenion.customize_string(judged_question)
                    id_judged_q = hit['id']
                    score_search = hit['score']
                    label = '0'
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
                    raw_file.write(origin_question + separator + judged_question + separator + label + '\n')
    raw_file.close()


raw_to_file(strict=False, tokenize=True, separator='\t')


>>>>>>> parent of 87789ce... more judged
def evaluate_classify_model():
    test_df = pd.read_csv('data/test-more-info.txt',
                          sep='\t',
                          header=None,
                          names=['id', 'origin_q', 'compare_q', 'label', 'score_elastic'],
                          )
    test_df['predict'] = 0.01

    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')
    classify_model = load_model('model/simple_classify_model.h5')

    for index, row in test_df.iterrows():
        origin_q = row['origin_q']
        compare_q = row['compare_q']

        origin_q_vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(origin_q, format='text')))
        compare_q_vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(compare_q, format='text')))
        concat_vector = np.concatenate((origin_q_vector, compare_q_vector))
        arr_wraper = np.array([concat_vector])

        test_df.at[index, 'predict'] = classify_model.predict(arr_wraper)[0][0]

    test = test_df.loc[test_df['id'] == 22972022]
    test_sort = test.sort_values(by='predict', ascending=False).reset_index(drop=True)

    id_queries = []
    for id_query in test_df['id']:
        if id_query not in id_queries:
            id_queries.append(id_query)

    mAP_df = pd.DataFrame(data=id_queries, columns=['id'])

    score_AP_model_alls = []
    score_AP_model_top10 = []
    score_AP_elastic_alls = []
    score_AP_elastic_top10 = []
    for id_query in mAP_df['id']:
        group_id = test_df.loc[test_df['id'] == id_query]

        # Caculate mAP model
        group_predict_sort = group_id.sort_values(
            by='predict',
            ascending=False).reset_index(drop=True)

        AP_model_all = convenion.caculate_AP(group_predict_sort['label'])
        AP_model_top10 = convenion.caculate_AP(group_predict_sort['label'][:10])

        score_AP_model_alls.append(AP_model_all)
        score_AP_model_top10.append(AP_model_top10)

        # Caculate mAP elastic search
        group_elastic_sort = group_id.sort_values(
            by='score_elastic',
            ascending=False).reset_index(drop=True)
        AP_elastic_all = convenion.caculate_AP(group_elastic_sort['label'])
        AP_elastic_top10 = convenion.caculate_AP(group_elastic_sort['label'][:10])

        score_AP_elastic_alls.append(AP_elastic_all)
        score_AP_elastic_top10.append(AP_elastic_top10)

    mAP_df['AP_model_all'] = score_AP_model_alls
    mAP_df['AP_model_top10'] = score_AP_model_top10

    mAP_df['AP_elastic_all'] = score_AP_elastic_alls
    mAP_df['AP_elastic_top10'] = score_AP_elastic_top10

    print('mAP elastic all: ', sum(score_AP_elastic_alls) / len(score_AP_elastic_alls))
    print('mAP model all: ', sum(score_AP_model_alls) / len(score_AP_model_alls))
    print('mAP elastic top10: ', sum(score_AP_elastic_top10) / len(score_AP_elastic_top10))
    print('mAP model top10: ', sum(score_AP_model_top10) / len(score_AP_model_top10))

    return mAP_df


# mAP_df = evaluate_classify_model()
