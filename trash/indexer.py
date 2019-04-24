import os.path
import json_lines
import convenion
from whoosh.index import *
from whoosh.fields import *
from whoosh.analysis import *


PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'


def index_basic():
    # Use scoring method BM25F
    print('Indexing basic...')
    schema = Schema(id=ID(stored=True),
                    question=STORED, answer=STORED,
                    question_custom=TEXT(stored=True),
                    answer_custom=TEXT(stored=True))

    if not os.path.exists('index_basic'):
        os.mkdir('index_basic')
    ix = create_in('index_basic', schema)
    writer = ix.writer()

    with open(PATH_QUESTION_ANSWER, 'r') as f:
        for qa in json_lines.reader(f):
            if not convenion.is_valid_qa(qa):
                continue
            question = qa['question']
            answer = qa['answer']
            question_custom = convenion.customize_and_remove_stopword(qa['question'])
            answer_custom = convenion.customize_and_remove_stopword(qa['answer'])
            print(question_custom)
            print(answer_custom)
            writer.add_document(id=qa['id_cmt'],
                                question=question, answer=answer,
                                question_custom=question_custom, answer_custom=answer_custom)
        print('Commit basic...')
        writer.commit()


def index_ngram():
    print('Indexing ngram...')
    schema = Schema(id=ID(stored=True), question=NGRAM(minsize=2, maxsize=7),
                    answer=NGRAM(minsize=2, maxsize=7))
    if not os.path.exists('index_ngram'):
        os.mkdir('index_ngram')
    ix = create_in('index_ngram', schema)
    writer = ix.writer()
    with open(PATH_QUESTION_ANSWER, 'r') as f:
        for qa in json_lines.reader(f):
            # print(qa['question'])
            # print(qa['answer'])
            # print('\n')
            if not convenion.is_valid_qa(qa):
                continue
            question = convenion.customize_and_remove_stopword(qa['question'])
            answer = convenion.customize_and_remove_stopword(qa['answer'])
            writer.add_document(id=qa['id_cmt'], question=question, answer=answer)
        print('Commit ngram...')
        writer.commit()


def index_ngram_word():
    print('Indexing ngram word...')
    schema = Schema(id=ID(stored=True),
                    question=NGRAMWORDS(minsize=2, maxsize=7, tokenizer=SpaceSeparatedTokenizer()),
                    answer=NGRAMWORDS(minsize=2, maxsize=7, tokenizer=SpaceSeparatedTokenizer()))
    if not os.path.exists('index_ngram_word'):
        os.mkdir('index_ngram_word')
    ix = create_in('index_ngram_word', schema)
    writer = ix.writer()
    with open(PATH_QUESTION_ANSWER, 'r') as f:
        for qa in json_lines.reader(f):
            # print(qa['question'])
            # print(qa['answer'])
            # print('\n')
            if not convenion.is_valid_qa(qa):
                continue
            question = convenion.customize_and_remove_stopword(qa['question'])
            answer = convenion.customize_and_remove_stopword(qa['answer'])
            writer.add_document(id=qa['id_cmt'], question=question, answer=answer)
        print('Commit ngram word...')
        writer.commit()


index_basic()

# print(get_stopwords())
