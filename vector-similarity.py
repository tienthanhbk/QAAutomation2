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


PATH_DATA_TRAIN = 'data/pool1/raw/train.txt'
PATH_DATA_DEV = 'data/pool1/raw/dev.txt'
PATH_DATA_TEST = 'data/test_data/raw/test.txt'

vector_dimention = 200

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
        self.result = K.mean(left * right, axis=-1, keepdims=True)
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def get_model():
    org_q_input = Input(shape=(vector_dimention,))
    related_q_input = Input(shape=(vector_dimention,))

    # output = ManDist()([org_q_input, related_q_input])
    # output = EuclidDist()([org_q_input, related_q_input])
    output = Cosine()([org_q_input, related_q_input])
    # output = dot([org_q_input, related_q_input], axes=-1, normalize=True) # Like cosine similarity

    training_model = Model(inputs=[org_q_input, related_q_input],
                           outputs=output,
                           name='training_model')

    # training_model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #
    # # training_model.compile(loss=contrastive_loss, optimizer=Adam())
    #
    print(training_model.summary())
    return training_model


def test():
    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')

    model = get_model()

    test_data_df = get_and_preprocess_data(PATH_DATA_TRAIN, separator='\t', more_info=False)

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values
    test_label_list = test_data_df['label'].astype('int')
    # test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
    #                                                                                padding=True, maxlen=60)
    input1s = []
    input2s = []
    for question in test_org_q_list:
        vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(question, format='text')))
        input1s.append(vector)

    for question in test_related_q_list:
        vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(question, format='text')))
        input2s.append(vector)

    predictions = model.predict([input1s, input2s])
    # predictions = np.random.rand(500)
    # predictions = np.zeros(2000)
    # print(predictions)

    # test_data_df['predict'] = predictions

    MAP, MRR = map_score(test_org_q_list, test_related_q_list, predictions, test_label_list)

    print("MAP: ", MAP)
    print("MRR: ", MRR)
    # mAP_df = classifier.caculate_map_queries(test_data_df)
    # mAP_df = mAP_df.sort_values(by=['id'])
    # return mAP_df


# PATH_VOCAB = 'data/word-vector/vocab_used.txt'
# vocab_df = pd.read_csv(PATH_VOCAB, sep='\t', index_col=1, header=None, names=['onehot'])
#
# # Adding 1 to onehot, considering 0 is padding value for one-hot vector
# vocab_df['onehot'] += 1
# # vocab_df.index.name = 'word'
# vocab_df.loc['<PAD>'] = 0
#
# vocab_df = vocab_df.sort_values(by=['onehot'])

# get_model(vocab_df)
# train(vocab_df)
# mAP_df = test(vocab_df)
test()