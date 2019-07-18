import os.path
import json_lines
import numpy as np
import re
from underthesea import word_tokenize
from keras.callbacks import Callback
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

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


def simple_customize_string(string):
    string = string.lower()
    string = string.replace('\xa0', ' ') \
        .replace('\r', ' ').replace('\n', ' ') \
        .strip()
    string = re.sub('\s+', ' ', string).strip()
    return string


def customize_string(string):
    if not isinstance(string, str):
        return string

    string = string.lower()
    string = string.replace('\xa0', ' ') \
        .replace('.', ' ').replace(',', ' ') \
        .replace('\r', ' ').replace('\n', ' ') \
        .replace('?', ' ').replace('!', ' ') \
        .replace('/', ' ').replace('-', '_') \
        .replace(':', ' ') \
        .strip()
    # string = re.sub(r'\bcmt\b', 'chứng minh thư', string)
    # string = re.sub(r'\bshk\b', 'sổ hộ khẩu', string)
    # string = re.sub(r'\bđt\b', 'điện thoại', string)
    # string = re.sub(r'\bdt\b', 'điện thoại', string)
    # string = re.sub(r'\bdc\b', 'được', string)
    # string = re.sub(r'\bdk\b', 'được', string)
    # string = re.sub(r'\bđk\b', 'được', string)
    # string = re.sub(r'\bđc\b', 'được', string)
    # string = re.sub(r'\bnhiu\b', 'nhiêu', string)
    # string = re.sub(r'\bbn\b', 'bao nhiêu', string)
    # string = re.sub(r'\bbnhieu\b', 'bao nhiêu', string)
    # string = re.sub(r'\bk\b', ' không', string)
    # string = re.sub(r'\bsp\b', 'sản phẩm', string)
    # string = re.sub(r'\blác\b', 'lag', string)
    # string = re.sub(r'\b0d\b', 'không đồng', string)
    # string = re.sub(r'\b0đ\b', 'không đồng', string)
    # string = re.sub(r'\b0 d\b', 'không đồng', string)
    # string = re.sub(r'\b0 đ\b', 'không đồng', string)
    # string = re.sub(r'\b12\b', ' mười_hai ', string)
    # string = re.sub(r'\b11\b', ' mười_một ', string)
    # string = re.sub(r'\b10\b', ' mười', string)
    # string = re.sub(r'\b9\b', ' chín', string)
    # string = re.sub(r'\bngắc\b', 'ngắt', string)
    # string = re.sub(r'\bsetting\b', 'cấu hình', string)
    # string = re.sub(r'\bmax\b', 'cao nhất', string)
    # string = re.sub(r'\bbóc hộp\b', 'mói', string)
    # string = re.sub(r'\bmở hộp\b', 'mới', string)
    # string = re.sub(r'\bhđh\b', 'hệ điều hành', string)
    # string = re.sub(r'\biphon\b', 'iphone', string)
    # string = re.sub(r'\bip\b', 'iphone', string)
    # string = re.sub(r'\bios11\b', 'ios mười_một', string)
    # string = re.sub(r'\bios10\b', 'ios mười', string)
    # string = re.sub(r'\bios9\b', 'ios chín', string)
    # string = re.sub(r'\bios12\b', 'ios mười_hai', string)
    # string = re.sub(r'\b10%\b', 'mười phần_trăm', string)
    # string = re.sub(r'\b15%\b', 'mười_năm phần_trăm', string)
    # string = re.sub(r'\b20%\b', 'hai_mươi phần_trăm', string)
    # string = re.sub(r'\b25%\b', 'hai_năm phần_trăm', string)
    # string = re.sub(r'\b30%\b', 'ba_mươi phần_trăm', string)
    # string = re.sub(r'\b35%\b', 'ba_năm phần_trăm', string)
    # string = re.sub(r'\b40%\b', 'bốn_mươi phần_trăm', string)
    # string = re.sub(r'\b50%\b', 'năm_mươi phần_trăm', string)
    # string = re.sub(r'\b60%\b', 'sáu_mưoi phần_trăm', string)
    # string = re.sub(r'\b20\b', 'hai_mươi', string)
    # string = re.sub(r'\b30\b', 'ba_mươi', string)
    # string = re.sub(r'\b40\b', 'bốn_mươi', string)
    # string = re.sub(r'\b50\b', 'năm_mươi', string)
    # string = re.sub(r'\b60\b', 'sáu_mươi', string)
    # string = re.sub(r'\b5\b', 'năm', string)
    # string = re.sub(r'\b0d\b', 'không trả trước', string)
    # string = re.sub(r'\b0%\b', 'không lãi suất', string)
    # string = re.sub(r'\b0 %\b', 'không lãi suất', string)
    # string = re.sub(r'\b0đ\b', 'không trả trước', string)
    # string = re.sub(r'\b0\b', 'không', string)
    # string = re.sub(r'%', ' phần_trăm ', string)
    # string = re.sub(r'\bsv\b', 'sinh viên', string)
    # string = re.sub(r'\btrk\b', 'trước', string)
    # string = re.sub(r'\bgplx\b', 'giấy phép lái xe', string)
    # string = re.sub(r'\bms\b', 'mới', string)
    # string = re.sub(r'\bh\b', 'giờ', string)
    # string = re.sub(r'\bmini chat\b', 'chat bong bóng', string)
    # string = re.sub(r'\bmini chát\b', 'chat bong bóng', string)
    # string = re.sub(r'\bnợ sấu\b', 'nợ xấu', string)
    # string = re.sub(r'\bgiấy cm\b', 'giấy chứng minh thư', string)
    # string = re.sub(r'\bnge\b', 'nghe', string)
    # string = re.sub(r'\bhsơ\b', 'hồ sơ', string)
    # string = re.sub(r'\bwf\b', 'wifi', string)
    # string = re.sub(r'\bonl\b', 'online', string)

    string = re.sub('\s+', ' ', string).strip()
    return word_tokenize(string, format="text")


# def get_stopwords():
#     with open(PATH_TO_STOPWORDS, 'r') as f:
#         return f.read().splitlines()


# def remove_stopword(string):
#     arr_stopword = get_stopwords()
#     arr_str = string.split(' ')
#     for str in arr_str:
#         if str in arr_stopword:
#             string = string.replace(str, '')
#     string = re.sub('\s+', ' ', string).strip()
#     return string


def customize_and_remove_stopword(string):
    string = customize_string(string)
    # string = remove_stopword(string)
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

# Methods use for all model
def get_word_vectors(path):
    print('Getting word vector...')
    # Return {'word1': ndarray[0.13, 0.44, ...], ...}
    word_vector = dict()
    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vector[word] = vector
        return word_vector


def get_and_preprocess_data(path, separator='\t\t\t', more_info=False):
    data_df = None
    if more_info:
        data_df = pd.read_csv(path, sep=separator, header=None, names=['org_q', 'related_q', 'label', 'id',
                                                                       'score_elastic'])
        # data_df['id'] = data_df['org_q']
    else:
        data_df = pd.read_csv(path, sep=separator, header=None, names=['org_q', 'related_q', 'label'])

    for row in data_df.itertuples():
        data_df.at[row.Index, 'org_q'] = customize_string(row.org_q)
        data_df.at[row.Index, 'related_q'] = customize_string(row.related_q)
    return data_df


def onehot_data(vocab_df, data_df, padding=True, maxlen=70):
    # Return ( org_q_onehot_list, related_q_onehot_list, label_list )
    # org_q_onehot_list: [ nparr[38, 85, 104, ...], ... ] ~ [ sequence_onehot_org_q, ... ]
    # [38, 85, 104, ...] represent a sentence; 38, 85, 54 represent a word (sequence_onehot_org_q)
    # related_q_onehot_list: [ nparr[14, 84, 98, ...], ... ] ~ [ sequence_onehot_related_q ]
    # label_list: [ 0, 1, 1, 0, 0, ... ]
    org_q_onehot_list = []
    related_q_onehot_list = []
    label_list = []

    for index, row in data_df.iterrows():
        sequence_onehot_org_q = np.array([])
        sequence_onehot_related_q = np.array([])
        # One-hot origin question
        for word in row['org_q'].split(' '):
            if word in vocab_df.index:
                sequence_onehot_org_q = np.append(sequence_onehot_org_q, vocab_df.loc[word]['onehot'])
                # sequence_onehot_org_q.append(int(vocab_df.loc[word]['onehot']))
            else:
                # If impact word not in vocab, ignore it
                continue

                # Orthercase: use value 0
                # sequence_onehot_org_q.append(0)

                # Other case: generate random number, it represent random word in vocab
                # rand_index = random.randint(0,len(vocab_df) - 1)
                # sequence_onehot_org_q = np.append(sequence_onehot_org_q, vocab_df.iloc[rand_index]['onehot'])

        org_q_onehot_list.append(sequence_onehot_org_q)

        # One-hot related question
        for word in row['related_q'].split(' '):
            if word in vocab_df.index:
                sequence_onehot_related_q = np.append(sequence_onehot_related_q, vocab_df.loc[word]['onehot'])
                # sequence_onehot_related_q.append(int(vocab_df.loc[word]['onehot']))
            else:
                continue
                # sequence_onehot_related_q.append(0)
        related_q_onehot_list.append(sequence_onehot_related_q)

        # Label
        label_list.append(int(row['label']))

    if padding:
        org_q_onehot_list = pad_sequences(org_q_onehot_list, maxlen=maxlen, padding='post', truncating='pre')
        related_q_onehot_list = pad_sequences(related_q_onehot_list, maxlen=maxlen, padding='post', truncating='pre')

    return org_q_onehot_list, related_q_onehot_list, label_list


def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
    # s1s_dev: ['abc xyz bla bla bla?', ...]: List of origin question
    # s2s_dev: ['abc xyz bla bla bla?', ...]: List of related question, respective order with s1s_dev
    # y_pred: [0.12, 0,78, ...]: Predictions of origin - related questions, respective order with s1s_dev and s2s_dev
    # labels_dev: [0, 1, ...]: Marsked labed of origin - related questions, respective order with s1s_dev and s2s_dev
    QA_pairs = {}
    for i in range(len(s1s_dev)):
        # pred = y_pred[i][1]
        pred = y_pred[i]

        s1 = str(s1s_dev[i])
        s2 = str(s2s_dev[i])
        if s1 in QA_pairs:
            QA_pairs[s1].append((s2, labels_dev[i], pred))
        else:
            QA_pairs[s1] = [(s2, labels_dev[i], pred)]

    MAP, MRR = 0, 0
    num_q = len(QA_pairs.keys())
    for s1 in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[s1] = sorted(
            QA_pairs[s1], key=lambda x: x[-1], reverse=True)

        for idx, (s2, label, _) in enumerate(QA_pairs[s1]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)
        if p == 0:
            AP = 0
        else:
            AP /= p
        MAP += AP
    MAP /= num_q
    MRR /= num_q
    return MAP, MRR


class AnSelCB(Callback):
    # def __init__(self, val_q, val_s, y, inputs):
    #     super().__init__()
    #     self.val_q = val_q
    #     self.val_s = val_s
    #     self.val_y = y
    #     self.val_inputs = inputs

    def __init__(self, val_data, train_data=None, test_data=None):
        super().__init__()
        self.val_q = val_data[0]
        self.val_s = val_data[1]
        self.val_y = val_data[2]
        self.val_inputs = val_data[3]

        self.train_q = None
        self.train_s = None
        self.train_y = None
        self.train_inputs = None
        if train_data is not None:
            self.train_q = train_data[0]
            self.train_s = train_data[1]
            self.train_y = train_data[2]
            self.train_inputs = train_data[3]

        self.test_q = None
        self.test_s = None
        self.test_y = None
        self.test_inputs = None
        if test_data is not None:
            self.test_q = test_data[0]
            self.test_s = test_data[1]
            self.test_y = test_data[2]
            self.test_inputs = test_data[3]

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.val_inputs)
        val_map__, val_mrr__ = map_score(self.val_q, self.val_s, val_pred, self.val_y)
        print('val MRR %f; val MAP %f' % (val_mrr__, val_map__))
        logs['val_mrr'] = val_mrr__
        logs['val_map'] = val_map__

        if self.train_inputs is not None:
            train_pred = self.model.predict(self.train_inputs)
            train_map__, train_mrr__ = map_score(self.train_q, self.train_s, train_pred, self.train_y)
            print('train MRR %f; train MAP %f' % (train_mrr__, train_map__))
            logs['train_mrr'] = train_mrr__
            logs['train_map'] = train_map__

        if self.test_inputs is not None:
            test_pred = self.model.predict(self.test_inputs)
            test_map__, test_mrr__ = map_score(self.test_q, self.test_s, test_pred, self.test_y)
            print('test MRR %f; test MAP %f' % (test_mrr__, test_map__))
            logs['test_mrr'] = test_mrr__
            logs['test_map'] = test_map__

print(customize_string(" Cám ơn\r\nChính sách bảo hành 1 đổi 1 :"))
