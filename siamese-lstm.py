import pandas as pd
import numpy as np
from underthesea import word_tokenize
from keras.layers import LSTM, Input, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization, SimpleRNN, Layer
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras import backend as K
import convenion

import re
import random
import json

import classifier

# @misc{word2vecvn_2016,
#     author = {Xuan-Son Vu},
#     title = {Pre-trained Word2Vec models for Vietnamese},
#     year = {2016},
#     howpublished = {\url{https://github.com/sonvx/word2vecVN}},
#     note = {commit xxxxxxx}
# }


PATH_DATA_TRAIN = 'data/pool1/raw/train.txt'
PATH_DATA_DEV = 'data/pool1/raw/dev.txt'
PATH_DATA_TEST = 'data/test_data/raw/test-moreinfo.txt'

PATH_DATA_TEST_SMALL = 'data/old_data/train-small.txt'
PATH_WORD_VECTOR = 'data/word-vector/vectors_baomoi.txt'
PATH_VOCAB = 'data/word-vector/vocab_used.txt'
# wordvector_dims = 300
maxlen_input = 60

num_units = 128


def customize_string(string):
    return convenion.customize_string(string)


def get_word_vectors():
    print('Getting word vector...')
    # Return {'word1': ndarray[0.13, 0.44, ...], ...}
    word_vector = dict()
    with open(PATH_WORD_VECTOR) as f:
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


class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.2
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * (1.0 - y_pred) + (1.0 - y_true) * K.maximum(0.0, y_pred - margin))


def get_model(vocab_df):
    # load the whole words embedding into memory
    word_vector = get_word_vectors()
    wordvector_dims = len(next(iter(word_vector.values())))

    num_vector = 0
    num_nonvector = 0

    # Create a weight matrix for words in vocab
    # Row i is vector for word indexed i in vocab
    # words = vocab_df.index.values
    # when one-hot word, 0 is padding value and not have in vocab
    embedding_weights = np.zeros((len(vocab_df), wordvector_dims))
    for word, row in vocab_df.iterrows():
        if word in word_vector:
            embedding_weights[row['onehot']] = word_vector[word]
            print('have word vector')
            num_vector += 1
        else:
            print('dont have word vector')
            embedding_weights[row['onehot']] = np.random.uniform(-0.25, 0.25, wordvector_dims).astype('float32')
            num_nonvector += 1

    print('Word have vector: ', num_vector)
    print('Word with random vector: ', num_nonvector)

    org_q_input = Input(shape=(maxlen_input,))
    related_q_input = Input(shape=(maxlen_input,))

    embedding = Embedding(input_dim=len(vocab_df),
                          output_dim=wordvector_dims,
                          weights=[embedding_weights],
                          trainable=True,
                          mask_zero=False)

    org_q_embedding = embedding(org_q_input)
    # org_q_embedding = Dropout(0.5)(org_q_embedding)

    related_q_embedding = embedding(related_q_input)
    # related_q_embedding = Dropout(0.5)(related_q_embedding)

    shared_lstm = LSTM(units=num_units, return_sequences=False)

    lstm_output_1 = shared_lstm(org_q_embedding)
    lstm_output_2 = shared_lstm(related_q_embedding)



    output = ManDist()([lstm_output_1, lstm_output_2])
    # concat = concatenate([lstm_output_1, lstm_output_2])
    # output = Dense(2, activation='softmax')(concat)
    output = Dropout(0.5)(output)

    training_model = Model(inputs=[org_q_input, related_q_input],
                           outputs=output,
                           name='training_model')
    # opt = Adam(lr=0.001)
    # training_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    training_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # training_model.compile(loss=contrastive_loss, optimizer=Adam())

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
                     ModelCheckpoint('siameselstm-0509-{epoch:02d}-{val_map:.2f}.h5', monitor='val_map',
                                     verbose=1,
                                     save_best_only=True, mode='max'),
                     EarlyStopping(monitor='val_map', mode='max', patience=20)]

    model = get_model(vocab_df)

    Y = np.array(train_label_list)

    model.fit(
        [train_org_q_onehot_list, train_related_q_onehot_list],
        Y,
        epochs=15,
        batch_size=32,
        validation_data=([dev_org_q_onehot_list, dev_related_q_onehot_list], dev_label_list),
        verbose=2
    )

    model.fit(
        [train_org_q_onehot_list, train_related_q_onehot_list],
        Y,
        epochs=100,
        batch_size=32,
        validation_data=([dev_org_q_onehot_list, dev_related_q_onehot_list], dev_label_list),
        verbose=2,
        callbacks=callback_list
    )

    history = model.history.history
    print(history)
    with open('siamese-lstm-pool1-0509.json', 'w+') as fp:
        json.dump(history, fp)


def test(vocab_df):
    model = get_model(vocab_df)
    model.load_weights('siameselstm-0509-21-0.69.h5')

    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t', more_info=True)

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values

    test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
                                                                                   padding=True, maxlen=maxlen_input)

    predictions = model.predict([test_org_q_onehot_list, test_related_q_onehot_list])
    # predictions = np.random.rand(450)

    test_data_df['predict'] = predictions

    MAP, MRR = map_score(test_org_q_list, test_related_q_list, predictions, test_label_list)

    print("MAP: ", MAP)
    print("MRR: ", MRR)
    mAP_df = classifier.caculate_map_queries(test_data_df)
    return mAP_df

vocab_df = pd.read_csv(PATH_VOCAB, sep='\t', index_col=1, header=None, names=['onehot'])

# Adding 1 to onehot, considering 0 is padding value for one-hot vector
vocab_df['onehot'] += 1
# vocab_df.index.name = 'word'
vocab_df.loc['<PAD>'] = 0

vocab_df = vocab_df.sort_values(by=['onehot'])

# get_model(vocab_df)
# train(vocab_df)
mAP_df = test(vocab_df)
