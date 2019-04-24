import os.path
import json_lines
from whoosh.index import *
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.scoring import *
import convenion

INDEX_PATH_BASIC = 'index_basic'
INDEX_PATH_NGRAM = 'index_ngram'
INDEX_PATH_NGRAM_WORD = 'index_ngram_word'


def search_basic(**kwargs):
    ix = open_dir(INDEX_PATH_BASIC)

    with ix.searcher() as searcher:
        lexicon = searcher.lexicon('answer_custom')
        print('len(lexicon) basic: ', len(list(lexicon)))
        # for vocab in lexicon:
        #     print(vocab.decode("utf-8", "replace"))
        parser = MultifieldParser(['question_custom', 'answer_custom'], ix.schema)
        str_query = convenion.customize_and_remove_stopword(kwargs['query'])
        print(str_query)
        myquery = parser.parse(str_query)

        results = searcher.search(myquery, limit=None)
        for result in results:
            yield result['id']


def search_ngram(**kwargs):
    ix = open_dir(INDEX_PATH_NGRAM)

    with ix.searcher() as searcher:
        lexicon = searcher.lexicon('answer')
        print('len(lexicon) ngram: ', len(list(lexicon)))
        parser = MultifieldParser(['question', 'answer'], ix.schema)

        myquery = parser.parse(kwargs['query'])

        results = searcher.search(myquery, limit=None)
        for result in results:
            yield result['id']


def search_ngram_word(**kwargs):
    ix = open_dir(INDEX_PATH_NGRAM_WORD)

    with ix.searcher() as searcher:
        lexicon = searcher.lexicon('answer')
        print('len(lexicon) ngram word: ', len(list(lexicon)))
        parser = MultifieldParser(['question', 'answer'], ix.schema)

        myquery = parser.parse(kwargs['query'])

        results = searcher.search(myquery, limit=None)
        for result in results:
            yield result['id']


def search_ngram_tfidf(**kwargs):
    ix = open_dir(INDEX_PATH_BASIC)

    with ix.searcher(weighting=TF_IDF) as searcher:
        parser = MultifieldParser(['question', 'answer'], ix.schema)

        myquery = parser.parse(kwargs['query'])

        results = searcher.search(myquery, limit=None)
        for result in results:
            yield result['id']


def search_frequency(**kwargs):
    ix = open_dir(INDEX_PATH_BASIC)

    with ix.searcher(weighting=Frequency) as searcher:
        parser = MultifieldParser(['question', 'answer'], ix.schema)

        # myquery = parser.parse(kwargs['query'])
        arr_term = []
        for word in kwargs['query'].split(' '):
            arr_term.append(Or([Term('question', word), Term('answer', word)]))
        myquery = And(arr_term)
        results = searcher.search(myquery, limit=None)
        for result in results:

            yield result['id']


def get_result_total(**kwargs):
    query = kwargs['query']

    results_basic = list(search_basic(query=query))
    results_ngram = list(search_ngram(query=query))
    results_ngram_word = list(search_ngram_word(query=query))
    results_tfidf = list(search_ngram_tfidf(query=query))
    results_frequency = list(search_frequency(query=query))

    print('results_basic len: ', len(results_basic))
    print('results_ngram len: ', len(results_ngram))
    print('results_ngram_word len: ', len(results_ngram_word))
    print('results_tfidf len: ', len(results_tfidf))
    print('results_frequency len: ', len(results_frequency))

    results_total = convenion.union_multi_arr(results_basic, results_ngram, results_ngram_word, results_tfidf)
    print('results_total len: ', len(results_total))

    print(results_total)
    for qa in convenion.get_qa_by_id(results_total):
        print(qa)


def get_lexicon_info(**kwargs):
    path_to_index = kwargs['path_to_index']
    ix = open_dir(path_to_index)

    with ix.searcher() as searcher:
        lexicon = searcher.lexicon('question_custom')
        print('len(lexicon) basic: ', len(list(lexicon)))
        # for vocab in lexicon:
        #     print(vocab.decode("utf-8", "replace"))


get_lexicon_info(path_to_index=INDEX_PATH_BASIC)

results_basic = list(search_basic(query='làm sao để tắt bluetooth a'))
print('results_basic len: ', len(results_basic))
for qa in convenion.get_qa_by_id(results_basic):
    print(qa)
