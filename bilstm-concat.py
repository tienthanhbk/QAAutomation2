import pandas as pd
import numpy as np
from underthesea import word_tokenize
from keras.layers import LSTM, Input, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization, SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import re
import random
import json


PATH_DATA_TRAIN = 'elastic/judged/pool2/split1/train.txt'
PATH_DATA_DEV = 'elastic/judged/pool2/split1/dev.txt'
PATH_DATA_TEST = 'elastic/judged/pool2/split1/test.txt'

PATH_DATA_TEST_SMALL = 'data/old_data/train-small.txt'
PATH_WORD_VECTOR = 'data/word-vector/vectors.txt'
PATH_VOCAB = 'data/word-vector/vocab_used.txt'
wordvector_dims = 200
maxlen_input = 150


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


def get_word_vectors():
    # Return {'word1': ndarray[0.13, 0.44, ...], ...}
    word_vector = dict()
    with open(PATH_WORD_VECTOR) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vector[word] = vector
        return word_vector


def get_and_preprocess_data(path, separator='\t\t\t'):
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
        org_q_onehot_list = pad_sequences(org_q_onehot_list, maxlen=maxlen, padding='post', truncating='post')
        related_q_onehot_list = pad_sequences(related_q_onehot_list, maxlen=maxlen, padding='post', truncating='post')

    return org_q_onehot_list, related_q_onehot_list, label_list


def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
    # s1s_dev: ['abc xyz bla bla bla?', ...]: List of origin question
    # s2s_dev: ['abc xyz bla bla bla?', ...]: List of related question, respective order with s1s_dev
    # y_pred: [0.12, 0,78, ...]: Predictions of origin - related questions, respective order with s1s_dev and s2s_dev
    # labels_dev: [0, 1, ...]: Marsked labed of origin - related questions, respective order with s1s_dev and s2s_dev
    QA_pairs = {}
    for i in range(len(s1s_dev)):
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


def get_model(vocab_df):
    # load the whole words embedding into memory
    word_vector = get_word_vectors()

    # Create a weight matrix for words in vocab
    # Row i is vector for word indexed i in vocab
    # words = vocab_df.index.values
    # when one-hot word, 0 is padding value and not have in vocab
    embedding_weights = np.zeros((len(vocab_df) + 1, wordvector_dims), dtype=float)
    for word, row in vocab_df.iterrows():
        embedding_weights[row['onehot']] = word_vector.get(word)

    org_q_input = Input(shape=(maxlen_input,))
    related_q_input = Input(shape=(maxlen_input,))

    embedding = Embedding(input_dim=len(vocab_df) + 1,
                          output_dim=wordvector_dims,
                          weights=[embedding_weights],
                          trainable=False,
                          mask_zero=True)

    org_q_embedding = embedding(org_q_input)
    related_q_embedding = embedding(related_q_input)

    bi_lstm_1 = Bidirectional(LSTM(units=4, return_sequences=False))(org_q_embedding)
    bi_lstm_2 = Bidirectional(LSTM(units=4, return_sequences=False))(related_q_embedding)
    # rnn1 = SimpleRNN(units=300, use_bias=True, return_sequences=False)(org_q_embedding)
    # rnn2 = SimpleRNN(units=300, use_bias=True, return_sequences=False)(org_q_embedding)

    q_concat = concatenate([bi_lstm_1, bi_lstm_2])
    # q_concat = concatenate([rnn1, rnn2])

    dense1 = Dense(64, activation='relu')(q_concat)
    prediction = Dense(1, activation='sigmoid')(dense1)

    training_model = Model(inputs=[org_q_input, related_q_input], outputs=prediction, name='training_model')
    opt = Adam(lr=0.001)
    training_model.compile(loss='binary_crossentropy', optimizer=opt)

    print(training_model.summary())
    return training_model


def train(vocab_df):
    train_data_df = get_and_preprocess_data(PATH_DATA_TRAIN, separator='\t')
    dev_data_df = get_and_preprocess_data(PATH_DATA_DEV, separator='\t')
    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t')

    train_org_q_onehot_list, train_related_q_onehot_list, train_label_list = onehot_data(vocab_df, train_data_df,
                                                                                         padding=True, maxlen=maxlen_input)
    dev_org_q_onehot_list, dev_related_q_onehot_list, dev_label_list = onehot_data(vocab_df, dev_data_df,
                                                                                   padding=True, maxlen=maxlen_input)
    test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
                                                                                      padding=True, maxlen=maxlen_input)

    dev_org_q_list = dev_data_df['org_q'].values
    dev_related_q_list = dev_data_df['related_q'].values

    train_org_q_list = train_data_df['org_q'].values
    train_related_q_list = train_data_df['related_q'].values

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values

    callback_val_data = [dev_org_q_list,
                         dev_related_q_list,
                         dev_label_list,
                         [dev_org_q_onehot_list, dev_related_q_onehot_list]]
    callback_train_data = [train_org_q_list,
                           train_related_q_list,
                           train_label_list,
                           [train_org_q_onehot_list, train_related_q_onehot_list]]
    callback_test_data = [test_org_q_list,
                          test_related_q_list,
                          test_label_list,
                          [test_org_q_onehot_list, test_related_q_onehot_list]]

    callback_list = [AnSelCB(callback_val_data, callback_train_data, callback_test_data),
                     ModelCheckpoint('biLSTM-{epoch:02d}-{val_map:.2f}.h5', monitor='val_map', verbose=1,
                                     save_best_only=True, mode='max'),
                     EarlyStopping(monitor='val_map', mode='max', patience=10)]

    model = get_model(vocab_df)

    Y = np.array(train_label_list)

    model.fit(
        [train_org_q_onehot_list, train_related_q_onehot_list],
        Y,
        epochs=50,
        batch_size=15,
        validation_data=([dev_org_q_onehot_list, dev_related_q_onehot_list], dev_label_list),
        verbose=1,
        callbacks=callback_list
    )

    history = model.history.history
    print(history)
    with open('history.json', 'w+') as fp:
        json.dump(history, fp)


def test(vocab_df):
    model = get_model(vocab_df)
    model.load_weights('biLSTM-16-0.97.h5')

    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t')

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values

    test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
                                                                                   padding=True, maxlen=maxlen_input)

    predictions = model.predict([test_org_q_onehot_list, test_related_q_onehot_list])

    MAP, MRR = map_score(test_org_q_list, test_related_q_list, predictions, test_label_list)

    print("MAP: ", MAP)
    print("MRR: ", MRR)


vocab_df = pd.read_csv(PATH_VOCAB, sep='\t', index_col=1, header=None, names=['onehot'])

# Adding 1 to onehot, considering 0 is padding value for one-hot vector
vocab_df['onehot'] += 1

# get_model(vocab_df)
train(vocab_df)
# test(vocab_df)