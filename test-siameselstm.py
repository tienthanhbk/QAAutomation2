import pandas as pd
import numpy as np
from underthesea import word_tokenize
from keras.layers import LSTM, Input, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization, SimpleRNN, Layer, dot
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

PATH_DATA_TRAIN = 'data/pool1/raw/train.txt'
PATH_DATA_DEV = 'data/pool1/raw/dev.txt'
PATH_DATA_TEST = 'data/test_data/raw/test.txt'
PATH_DATA_TEST_MORE_INFO = 'data/test_data/raw/test-moreinfo.txt'

PATH_WORD_VECTOR = 'data/word-vector/vectors_baomoi.txt'
PATH_VOCAB = 'data/word-vector/vocab_used.txt'
# wordvector_dims = 300
maxlen_input = 60

num_units = 256

from convenion import get_word_vectors, get_and_preprocess_data, onehot_data, map_score, AnSelCB


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


class EuclidDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(EuclidDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(EuclidDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class Cosine(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(Cosine, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(Cosine, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        left, right = x
        left = K.l2_normalize(left, axis=-1)
        right = K.l2_normalize(right, axis=-1)
        self.result = -K.mean(left * right, axis=-1, keepdims=True)
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def get_model(vocab_df):
    # load the whole words embedding into memory
    word_vector = get_word_vectors(PATH_WORD_VECTOR)
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
            num_vector += 1
        else:
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



    # output = ManDist()([lstm_output_1, lstm_output_2])
    output = EuclidDist()([lstm_output_1, lstm_output_2])
    # output = Cosine()([lstm_output_1, lstm_output_2])
    # output = dot([lstm_output_1, lstm_output_2], axes=-1, normalize=True) # Like cosine similarity

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


def test(vocab_df):
    model = get_model(vocab_df)
    model.load_weights('model/siamese lstm/euclid/memory units 256/siameselstm-0509-21-0.71.h5')

    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t', more_info=False)

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values

    test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
                                                                                   padding=True, maxlen=maxlen_input)

    predictions = model.predict([test_org_q_onehot_list, test_related_q_onehot_list])
    # predictions = np.random.rand(450)
    # predictions = np.zeros(300)

    test_data_df['predict'] = predictions

    MAP, MRR = map_score(test_org_q_list, test_related_q_list, predictions, test_label_list)

    print("MAP: ", MAP)
    print("MRR: ", MRR)
    # mAP_df = classifier.caculate_map_queries(test_data_df)
    # mAP_df = mAP_df.sort_values(by=['id'])
    # return mAP_df


vocab_df = pd.read_csv(PATH_VOCAB, sep='\t', index_col=1, header=None, names=['onehot'])

# Adding 1 to onehot, considering 0 is padding value for one-hot vector
vocab_df['onehot'] += 1
# vocab_df.index.name = 'word'
vocab_df.loc['<PAD>'] = 0

vocab_df = vocab_df.sort_values(by=['onehot'])

# get_model(vocab_df)
# train(vocab_df)
# mAP_df = test(vocab_df)
test(vocab_df)
