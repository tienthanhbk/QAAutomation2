import pandas as pd
import numpy as np
from keras.layers import LSTM, Input, Dense, Dropout, concatenate
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding


PATH_DATA_TRAIN = 'data/pool1/raw/train.txt'
PATH_DATA_DEV = 'data/pool1/raw/dev.txt'
PATH_DATA_TEST = 'data/test_data/raw/test.txt'
PATH_DATA_TEST_MORE_INFO = 'data/test_data/raw/test-moreinfo.txt'

PATH_WORD_VECTOR = 'data/word-vector/vectors_baomoi.txt'
PATH_VOCAB = 'data/word-vector/vocab_used.txt'
maxlen_input = 60

num_hidden_node = 64
num_units = 128

from convenion import get_word_vectors, get_and_preprocess_data, onehot_data, map_score, AnSelCB


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

    bi_lstm_1 = Bidirectional(LSTM(units=num_units, return_sequences=False))(org_q_embedding)
    bi_lstm_2 = Bidirectional(LSTM(units=num_units, return_sequences=False))(related_q_embedding)
    # rnn1 = SimpleRNN(units=300, use_bias=True, return_sequences=False)(org_q_embedding)
    # rnn2 = SimpleRNN(units=300, use_bias=True, return_sequences=False)(org_q_embedding)
    # shared_lstm = Bidirectional(LSTM(units=num_units, return_sequences=False))
    # shared_lstm = Dropout(0.5)(shared_lstm)
    # bi_lstm_1 = shared_lstm(org_q_embedding)
    # bi_lstm_2 = shared_lstm(related_q_embedding)

    # lstm_output_1 = bi_lstm_1(org_q_embedding)
    # lstm_output_2 = bi_lstm_2(related_q_embedding)

    q_concat = concatenate([bi_lstm_1, bi_lstm_2])
    # q_concat = Dropout(0.5)(q_concat)
    # q_concat = concatenate([rnn1, rnn2])

    dense1 = Dense(num_hidden_node, activation='relu')(q_concat)
    prediction = Dense(1, activation='sigmoid')(dense1)
    prediction = Dropout(0.5)(prediction)

    training_model = Model(inputs=[org_q_input, related_q_input], outputs=prediction, name='training_model')
    opt = Adam(lr=0.001)
    training_model.compile(loss='binary_crossentropy', optimizer='adam')

    print(training_model.summary())
    return training_model


def test(vocab_df):
    model = get_model(vocab_df)
    model.load_weights('model/bi-lstm-feature/hiddenlayer 64/memory units 128/biLSTM-15-0.68-128units-64nodes.h5')

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
# vocab_df.index.name = 'word'
vocab_df.loc['<PAD>'] = 0

vocab_df = vocab_df.sort_values(by=['onehot'])

# get_model(vocab_df)
test(vocab_df)
