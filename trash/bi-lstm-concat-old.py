import numpy as np
from underthesea import word_tokenize
from keras.layers import LSTM, Input, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import random
import nltk
import re

nltk.download('punkt')


# from google.colab import drive
# drive.mount('/content/driver/')

PATH_DATA_TRAIN = 'data/train.txt'
PATH_DATA_DEV = 'data/dev.txt'
PATH_DATA_TEST = 'data/test.txt'
PATH_WORD_VECTOR = 'data/word-vector/vectors.txt'
PATH_VOCAB = 'data/word-vector/vocab_used.txt'
wordvector_dims = 200


# Defined method
# Coppy from convenion file
def customize_string(string):
    # Return string
    replacer_arr = ['.', ',', '?', '\xa0', '\t']
    string = string.lower().replace('\xa0', ' ')\
        .replace('.', ' ').replace(',', ' ')\
        .replace('?', ' ').replace('!', ' ')\
        .replace('/', ' ').replace('-', '_') \
        .replace(':', ' ') \
        .strip()
    string = re.sub('\s+', ' ', string).strip()
    return word_tokenize(string, format="text")


def get_processed_data(FILE_PATH):
    # Return: [[org_q, related_q, label], ...]
    f = open(FILE_PATH, 'r')
    data_processed = []
    for line in f.readlines():
        line = line.strip()
        temp = line.split('\t\t\t')
        for i in range(2):
            temp[i] = customize_string(temp[i])
            # temp[i] = temp[i].lower()
            # temp[i] = word_tokenize(temp[i], format='text')
        data_processed.append(temp)
    f.close()
    print(data_processed[:10])
    return data_processed


def build_corpus(FILE_PATH):
    # Return: questions_origin: list of [org_q1, ...]
    # questions_related:[related_q1, ...]
    # labels: [1, 0, 0, 1, ...]
    data_processed = get_processed_data(FILE_PATH)
    questions_origin = []
    questions_related = []
    labels = []
    # for i in range(len(data_processed)):
    #     questions_origin.extend([data_processed[i][0]])
    #     questions_related.append([data_processed[i][1]])
    #     labels.append(int(data_processed[i][2]))
    for datum in data_processed:
        questions_origin.append(datum[0])
        questions_related.append(datum[1])
        labels.append(int(datum[2]))
    print(questions_origin)
    print(questions_related)
    print(labels)
    return questions_origin, questions_related, labels


def sentence_to_vec(sentence, vocab):
    splited_sentence = sentence.split(' ')
    result = np.zeros([len(splited_sentence), ], dtype=int)
    for i in range(len(splited_sentence)):
        if splited_sentence[i] in vocab:
            result[i] = get_index(splited_sentence[i], vocab)
        else:
            result[i] = random.randint(0, len(vocab))
    return result


def turn_to_vector(list_to_transform, vocab):
    # vocab_size = 44604
    # pad = 150
    encoded_list = [sentence_to_vec(str(d), vocab) for d in list_to_transform]
    padded_list = pad_sequences(encoded_list, maxlen=150, padding='post', truncating='post')
    return padded_list


def get_index(word, vocab):
    return vocab[word]


def create_vocab_with_index(model):
    with open(PATH_VOCAB, 'w+') as f:
        vocab = model.wv.vocab
        for key, _ in vocab.items():
            index = vocab[key].index
            f.write(str(index) + '\t' + key)
            f.write('\n')
    f.close()


def create_vocab_dict():
    vocab = {}
    with open(PATH_VOCAB, 'r') as f:
        for line in f.readlines():
            temp = line.split('\t')
            vocab[temp[1].strip()] = temp[0].strip()
    f.close()
    return vocab


def get_glove_vectors():
    g = dict()
    file_path = PATH_WORD_VECTOR

    with open(file_path, 'r') as f:
        for line in f.readlines():
            temp = line.split()
            word = temp[0]
            g[word] = np.array(temp[1:]).astype(float)
    return g


# Return 2D matrix that row i is
# wordvector of word indexed i in vocab
def embmatrix(g, vocab):
    non_vocab = 0
    embedding_weights = np.zeros((len(vocab) + 1, wordvector_dims), dtype=float)
    for word in vocab.keys():
        if word in g:
            embedding_weights[int(vocab[word]), :] = np.array(g[word])
        else:
            non_vocab += 1
            embedding_weights[int(vocab[word]), :] = np.random.uniform(-1, 1, wordvector_dims)
    print('non vocab: ', non_vocab)
    return embedding_weights


def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
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
    def __init__(self, val_q, val_s, y, inputs):
        self.val_q = val_q
        self.val_s = val_s
        self.val_y = y
        self.val_inputs = inputs

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        map__, mrr__ = map_score(self.val_q, self.val_s, pred, self.val_y)
        print('val MRR %f; val MAP %f' % (mrr__, map__))
        logs['mrr'] = mrr__
        logs['map'] = map__


# Main methods
def get_bilstm_model(vocab_size, vocab):
    enc_timesteps = 150
    dec_timesteps = 150
    # hidden_dim = 128

    question = Input(shape=(enc_timesteps,),
                     dtype='int32', name='question_base')
    answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')

    g = get_glove_vectors()
    weights = embmatrix(g, vocab)
    # When mask_zero=True, input_dim should be vocab_size + 1 and input_length is not fixed
    qa_embedding = Embedding(
        input_dim=vocab_size + 1,
        input_length=None,
        output_dim=weights.shape[1],
        mask_zero=True,
        weights=[weights])
    # qa_embedding.trainable = False
    # units: dimensionality of the output space
    bi_lstm1 = Bidirectional(LSTM(units=200, return_sequences=False))
    bi_lstm2 = Bidirectional(LSTM(units=200, return_sequences=False))

    question_embedding = qa_embedding(question)
    # question_embedding = Dropout(0.75)(question_embedding)
    question_enc_1 = bi_lstm1(question_embedding)
    # question_enc_1 = Dropout(0.75)(question_enc_1)
    # question_enc_1 = BatchNormalization()(question_enc_1)

    answer_embedding = qa_embedding(answer)
    # answer_embedding = Dropout(0.75)(answer_embedding)
    answer_enc_1 = bi_lstm2(answer_embedding)
    # answer_enc_1 = Dropout(0.75)(answer_enc_1)
    # answer_enc_1 = BatchNormalization()(answer_enc_1)

    qa_merged = concatenate([question_enc_1, answer_enc_1])
    dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0007))(qa_merged)
    # qa_merged = Dropout(0.75)(qa_merged)
    prediction = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
                      activity_regularizer=regularizers.l1(0.001))(dense1)

    # lstm_model = Model(name="bi_lstm", inputs=[
    #     question, answer], outputs=qa_merged)
    # output = lstm_model([question, answer])
    training_model = Model(
        inputs=[question, answer], outputs=prediction, name='training_model')
    opt = Adam(lr=0.0001)
    training_model.compile(loss='binary_crossentropy', optimizer=opt)
    return training_model


def train(vocab):
    vocab_len = len(vocab)

    training_model = get_bilstm_model(vocab_len, vocab)

    questions_origin, question_related, labels = build_corpus(PATH_DATA_TRAIN)

    q_origin_dev, q_related_dev, l_dev = build_corpus(PATH_DATA_DEV)

    # questions_origin, question_related = turn_to_vector(questions_origin, question_related, tok)
    # q_origin_dev_eb, q_related_dev_eb = turn_to_vector(q_origin_dev, q_related_dev, tok)
    questions_origin = turn_to_vector(questions_origin, vocab)
    question_related = turn_to_vector(question_related, vocab)
    q_origin_dev_eb = turn_to_vector(q_origin_dev, vocab)
    q_related_dev_eb = turn_to_vector(q_related_dev, vocab)
    Y = np.array(labels)
    callback_list = [AnSelCB(q_origin_dev, q_related_dev, l_dev, [q_origin_dev_eb, q_related_dev_eb]),
                     ModelCheckpoint('model_BiLSTMimprovement-{epoch:02d}-{map:.2f}.h5', monitor='map', verbose=1,
                                     save_best_only=True, mode='max'),
                     EarlyStopping(monitor='map', mode='max', patience=20)]

    training_model.fit(
        [questions_origin, question_related],
        Y,
        epochs=100,
        batch_size=100,
        validation_data=([q_origin_dev_eb, q_related_dev_eb], l_dev),
        verbose=1,
        callbacks=callback_list
    )
    training_model.summary()


def test_model(vocab):
    vocab_len = len(vocab)

    training_model = get_bilstm_model(vocab_len, vocab)
    training_model.load_weights('model/bi-word-vector/1biLSTM-baomoi-200units-08-0.28.h5')

    questions_origin, question_related, labels = build_corpus(PATH_DATA_TEST)
    print(len(questions_origin))

    questions_origin_eb = turn_to_vector(questions_origin, vocab)
    question_related_eb = turn_to_vector(question_related, vocab)

    # Y = np.array(labels)

    sims = training_model.predict([questions_origin_eb, question_related_eb])

    MAP, MRR = map_score(questions_origin, question_related, sims, labels)
    print("MAP: ", MAP)
    print("MRR: ", MRR)


# Run
vocab = create_vocab_dict()
vocab_len = len(vocab)
# training_model = get_bilstm_model(vocab_len, vocab)
# print(training_model.summary())
