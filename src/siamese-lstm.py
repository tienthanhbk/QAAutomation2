import pandas as pd
import numpy as np
from keras.layers import LSTM, Input, Dropout, Layer
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras import backend as K
import json

from src import classifier

# @misc{word2vecvn_2016,
#     author = {Xuan-Son Vu},
#     title = {Pre-trained Word2Vec models for Vietnamese},
#     year = {2016},
#     howpublished = {\url{https://github.com/sonvx/word2vecVN}},
#     note = {commit xxxxxxx}
# }


PATH_DATA_TRAIN = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/collection/raw/train.txt'
PATH_DATA_DEV = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/collection/raw/dev.txt'
PATH_DATA_TEST = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/test_data/raw/test.txt'
PATH_DATA_TEST_MORE_INFO = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/test_data/raw/test-moreinfo.txt'

PATH_WORD_VECTOR = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/word-vector/vectors_baomoi.txt'
PATH_VOCAB = '/Users/tienthanh/Projects/ML/QAAutomation/data/Vietnamese/word-vector/vocab_used.txt'

PATH_MODEL_TEST = '/Users/tienthanh/Projects/ML/QAAutomation/model/Vietnamese/siamese lstm/euclid/memory units 256/siameselstm-0509-21-0.71.h5'
# wordvector_dims = 300
maxlen_input = 60

num_units = 256

from src.convenion import get_word_vectors, get_and_preprocess_data, onehot_data, map_score, AnSelCB


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
                     EarlyStopping(monitor='val_map', mode='max', patience=10)]

    model = get_model(vocab_df)

    Y = np.array(train_label_list)

    # model.fit(
    #     [train_org_q_onehot_list, train_related_q_onehot_list],
    #     Y,
    #     epochs=15,
    #     batch_size=32,
    #     validation_data=([dev_org_q_onehot_list, dev_related_q_onehot_list], dev_label_list),
    #     verbose=2
    # )

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
    with open('test-pool1-0519.json', 'w+') as fp:
        json.dump(history, fp)


def test(vocab_df):
    model = get_model(vocab_df)
    model.load_weights(PATH_MODEL_TEST)

    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t', more_info=True)

    test_org_q_list = test_data_df['org_q'].values
    test_related_q_list = test_data_df['related_q'].values

    test_org_q_onehot_list, test_related_q_onehot_list, test_label_list = onehot_data(vocab_df, test_data_df,
                                                                                   padding=True, maxlen=maxlen_input)

    predictions = model.predict([test_org_q_onehot_list, test_related_q_onehot_list])

    test_data_df['predict'] = predictions

    MAP, MRR = map_score(test_org_q_list, test_related_q_list, predictions, test_label_list)

    print("MAP: ", MAP)
    print("MRR: ", MRR)
    mAP_df = classifier.caculate_map_queries(test_data_df)
    mAP_df = mAP_df.sort_values(by=['id'])
    return mAP_df

vocab_df = pd.read_csv(PATH_VOCAB, sep='\t', index_col=1, header=None, names=['onehot'])

# Adding 1 to onehot, considering 0 is padding value for one-hot vector
vocab_df['onehot'] += 1
# vocab_df.index.name = 'word'
vocab_df.loc['<PAD>'] = 0

vocab_df = vocab_df.sort_values(by=['onehot'])


model = get_model(vocab_df)
# train(vocab_df)
# mAP_df = test(vocab_df)
# test(vocab_df)
embeddings = model.layers[2].get_weights()[0]

words_embeddings = {w:embeddings[idx][0] for w, idx in vocab_df.iterrows()}
words_embeddings_df = pd.DataFrame.from_dict(words_embeddings, orient='index')
words_embeddings_df.to_csv('/Users/tienthanh/Projects/ML/QAAutomation/gensim/wv', sep=' ')

