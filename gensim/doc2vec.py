from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models import KeyedVectors
import smart_open
# import json_lines
from underthesea import word_tokenize
import jsonlines
import os
import collections
import random
from pyvi import ViTokenizer, ViPosTagger


# Base on https://blog.duyet.net/2017/10/doc2vec-trong-sentiment-analysis.html#.XJdzLOszbwc

# PATH_QA = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QA = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'


def read_corpus(fname):
    with jsonlines.open(fname) as reader:
        for line in reader:
            id_qa = line['id_cmt']
            question = line['question']
            answer = line['answer']
            print(question)
            print(answer)
            if id_qa is None or question is None:
                continue
            question = word_tokenize(question, format='text')
            answer = word_tokenize(answer, format='text')
            yield TaggedDocument(simple_preprocess(question), [id_qa + '_q'])
            yield TaggedDocument(simple_preprocess(answer), [id_qa + '_a'])


def train_model():
    train_corpus = list(read_corpus(PATH_QA))
    print('train corpus total sentences: ', len(train_corpus))

    print(train_corpus[:10])
    model = Doc2Vec(vector_size=200, min_count=2, epochs=50)
    # Build
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('gensim/model/question.d2v')
    print('Trained and saved')


def test_model():
    doc2vec_model = model = Doc2Vec.load('gensim/model/question.d2v')
    print(model.wv.most_similar('pin'))
    print(model.wv.most_similar('ip'))
    print(model.wv.most_similar('loa'))
    # print(model.infer_vector(['còn', 'hàng', 'không']))


def raw_my_vectors_to_file(model):
    word_vectors = model.wv

    word_vectors.word_vec('pin')
    with open('data/word-vector/vectors_baomoi.txt', 'w+') as f:
        vocab = model.wv.vocab
        for key, _ in vocab.items():
            word_vector = word_vectors.word_vec(key)
            wv_str = ''
            for num in word_vector:
                wv_str += str(num)
                wv_str += ' '
            f.write(key + ' ' + wv_str)
            f.write('\n')
        f.close()


def raw_vocab_with_index(model):
    with open('data/word-vector/vocab_baomoi.txt', 'w+') as f:
        vocab = model.wv.vocab
        for key, _ in vocab.items():
            index = vocab[key].index
            f.write(str(index) + '\t' + key)
            f.write('\n')
    f.close()

def raw_vectors_and_vocab():
    baomoi_model = KeyedVectors.load_word2vec_format('gensim/model/baomoi.model.bin', binary=True)

    raw_my_vectors_to_file(baomoi_model)
    raw_vocab_with_index(baomoi_model)
