from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
# import json_lines
from underthesea import word_tokenize
import jsonlines
from src import convenion

# Base on https://blog.duyet.net/2017/10/doc2vec-trong-sentiment-analysis.html#.XJdzLOszbwc

# PATH_QA = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QA = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'


def read_corpus(fname):
    with jsonlines.open(fname) as reader:
        for line in reader:
            id_qa = line['id_cmt']
            question = line['question']
            answer = line['answer']
            # print(question)
            # print(answer)

            if id_qa is None or question is None or answer is None:
                continue

            question = convenion.customize_string(question)
            answer = convenion.customize_string(answer)
            question = word_tokenize(question, format='text')
            answer = word_tokenize(answer, format='text')
            yield TaggedDocument(simple_preprocess(question), [id_qa + '_q'])
            yield TaggedDocument(simple_preprocess(answer), [id_qa + '_a'])


def train_model():
    train_corpus = list(read_corpus(PATH_QA))
    print('train corpus total sentences: ', len(train_corpus))

    print(train_corpus[:10])
    model = Doc2Vec(vector_size=200, min_count=1, epochs=50)
    # Build
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('/Users/tienthanh/Projects/ML/QAAutomation/gensim/model/question.d2v')
    print('Trained and saved')


def test_model():
    # model = Doc2Vec.load('/Users/tienthanh/Projects/ML/QAAutomation/gensim/model/question.d2v')
    # model.save_word2vec_format('/Users/tienthanh/Projects/ML/QAAutomation/gensim/test')
    model = KeyedVectors.load_word2vec_format('/Users/tienthanh/Projects/ML/QAAutomation/gensim/wv')
    return model
    # print(model.wv.most_similar('pin - sạc'))
    # print(model.wv.most_similar('ip'))
    # print(model.wv.most_similar('loa - âm thanh'))
    # print(model.infer_vector(['còn', 'hàng', 'không']))


def raw_my_vectors_to_file(model):
    word_vectors = model.wv

    # word_vectors.word_vec('pin - sạc')
    with open('/Users/tienthanh/Projects/ML/QAAutomation/data/word-vector/vectors.txt', 'w+') as f:
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
    with open('/Users/tienthanh/Projects/ML/QAAutomation/data/word-vector/vocab_used.txt', 'w+') as f:
        vocab = model.wv.vocab
        for key, _ in vocab.items():
            index = vocab[key].index
            f.write(str(index) + '\t' + key)
            f.write('\n')
    f.close()

def raw_vectors_and_vocab():
    # baomoi_model = KeyedVectors.load_word2vec_format('gensim/model/baomoi.model.bin', binary=True)
    my_model = Doc2Vec.load('/Users/tienthanh/Projects/ML/QAAutomation/gensim/model/question.d2v')

    raw_my_vectors_to_file(my_model)
    raw_vocab_with_index(my_model)


# train_model()
# raw_vectors_and_vocab()
model = test_model()

model.get_vector('anh')
model.similar_by_word('iphon')
model.most_similar()

from sklearn.decomposition import PCA
from matplotlib import pyplot

# X = model[model.wv.vocab]
# pca = PCA(n_components=3)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()

# train_corpus = list(read_corpus(PATH_QA))
#
# model_2 = Word2Vec(size=400, min_count=1)
# model_2.build_vocab(train_corpus)
# total_examples = model_2.corpus_count
# model = KeyedVectors.load_word2vec_format("/Users/tienthanh/Projects/ML/QAAutomation/gensim/wv", binary=False)
# model_2.build_vocab([list(model.vocab.keys())], update=True)
# model_2.intersect_word2vec_format("/Users/tienthanh/Projects/ML/QAAutomation/gensim/wv", binary=False, lockf=1.0)
# model_2.train(train_corpus, total_examples=total_examples, epochs=model_2.iter)
