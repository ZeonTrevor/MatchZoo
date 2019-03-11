# -*- coding=utf-8 -*-

from __future__ import print_function
import lucene
from java.nio.file import Paths
from java.io import StringReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import LMDirichletSimilarity, BM25Similarity
from org.apache.lucene.search.highlight import Highlighter, QueryScorer, SimpleFragmenter
from bs4 import BeautifulSoup
from sklearn.covariance import robust_covariance

from matchzoo.inputs.preprocess import *

import os
import gzip
import re
import collections
import sys
from tqdm import tqdm
sys.path.append('../common_utils')
# from IndexSearcher import get_lm_matched_docs
# from IndexSearcher import init
from Queue import Queue
from threading import Thread, Lock
from datetime import datetime
import multiprocessing


def filter_aol_queries(aol_data_dir, filtered_queries_dir):
    """
    filters navigational queries from the AOL query log in aol_data_dir
    and stores the filtered queries in filtered_queries_dir
    :param aol_data_dir:
    :param filtered_queries_dir:
    :return:
    """
    nav_query_substr = ['http', 'www.', '.com', '.net', '.org', '.edu']

    unique_queries = set()
    filtered_nav_queries = set()
    count_total_lines = 0
    for filename in os.listdir(aol_data_dir):
        print(filename)
        uniq_q_file = set()
        filtered_nav_q_file = set()
        with gzip.open(aol_data_dir + filename, 'rb') as f:
            count = 0
            for line in f:
                count = count + 1
                if count == 1:
                    continue

                line = line.strip().split("\t")
                query = line[1]
                unique_queries.add(query)
                uniq_q_file.add(query)
                if not any(substr in query for substr in nav_query_substr):
                    filtered_nav_queries.add(query)
                    filtered_nav_q_file.add(query)
            print('No. of lines read %d' % count)
            count_total_lines = count_total_lines + count
            percentage = len(filtered_nav_q_file) * 100 / len(uniq_q_file)
            print('No. of unique queries in file({}) : {}/{} {:.2f}'.format(filename, len(filtered_nav_q_file),
                                                                            len(uniq_q_file), percentage))

    print('Total no. of lines read %d' % count_total_lines)
    total_percentage = len(filtered_nav_queries) * 100 / len(unique_queries)
    print('Percentage of filtered queries in total: {}/{} {:.2f}'.format(len(filtered_nav_queries), len(unique_queries),
                                                                         total_percentage))
    with gzip.open(filtered_queries_dir + 'queries.txt.gz', 'w') as fout:
        for query in filtered_nav_queries:
            print(query, file=fout)


def filtered_queries_for_collection(filtered_queries_dir, collection_query_file):
    """
    Removes queries from the evaluation set in the training data query set which is stored
    in a directory
    :param filtered_queries_dir:
    :param collection_query_file:
    :return:
    """
    queries = []
    pattern = re.compile('([^\s\w]|_)+')
    # raw_queries = []
    with gzip.open(filtered_queries_dir + 'queries.txt.gz', 'rb') as f:
        for line in f:
            query = pattern.sub('', line)
            query = ' '.join(query.strip().split())
            queries.append(query)
            # raw_queries.append(line.strip())

    collection_queries = []
    with open(collection_query_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            query = " ".join(line[1:])
            collection_queries.append(query)

    print("Total nr. of queries %d " % len(queries))
    print("Total nr. of collection queries %d" % len(collection_queries))
    # for query, raw_query in zip(queries[:250], raw_queries[:250]):
    #     print("%s -> %s" % (raw_query, query))

    uniq_q = set(queries)
    uniq_collec_q = set(collection_queries)
    filtered_uniq_q = uniq_q.difference(uniq_collec_q)
    # uniq_collec_q_in_uniq_q = uniq_collec_q.difference(uniq_collec_q.difference(uniq_q))

    # print('len of collection queries in set of unique aol queries %d' % len(uniq_collec_q_in_uniq_q))
    print('len of unique queries %d' % len(uniq_q))
    print('len of filtered unique queries %d' % len(filtered_uniq_q))
    print('Difference after removing non-alphanum chars: %d' % (len(queries) - len(uniq_q)))
    print('Difference after removing queries in eval set: %d' % (len(uniq_q) - len(filtered_uniq_q)))

    filtered_uniq_q.remove('')
    print(len(filtered_uniq_q))
    filtered_uniq_q = list(filtered_uniq_q)

    with gzip.open(filtered_queries_dir + 'queries_cw_not_filtered.txt.gz', 'w') as fout:
        for id, query in enumerate(filtered_uniq_q):
            print(id, query, file=fout)


class MyThread(Thread):
    def __init__(self, tid, name, q, out_q):
        Thread.__init__(self)
        self.tid = tid
        self.name = name
        self.q = q
        self.out_q = out_q

    def run(self):
        print("Starting " + self.name)
        lucene.getVMEnv().attachCurrentThread()
        index = DirectoryReader.open(SimpleFSDirectory(Paths.get(robust_index_dir)))
        searcher = IndexSearcher(index)
        searcher.setSimilarity(BM25Similarity())
        analyzer = EnglishAnalyzer()
        qparser = QueryParser("contents", analyzer)
        # process_query(self.name, self.q, self.out_q, searcher, qparser)
        print("Exiting " + self.name)


def get_lm_matched_docs(query, searcher, qparser, topk=2000):
    """
    Fetches the topk documents given query, searcher and qparser and
    returns doc_id and score lists
    :param query:
    :param searcher:
    :param qparser:
    :param topk:
    :return:
    """
    #did_dict = {}
    dids = []
    scores = []
    query = qparser.parse(query)
    # searcher.setSimilarity(LMDirichletSimilarity())
    scoreDocs = searcher.search(query, topk).scoreDocs
    # print("Found %d document(s) that matched query '%s':" % (len(scoreDocs), query))

    for scoreDoc in scoreDocs:
        if len(dids) > 1000:
            break

        doc = searcher.doc(scoreDoc.doc)
        did = doc.get("id")

        if check_if_spam(did):
            continue
        #text = doc.get("raw")
        #did_dict[did] = {}
        #did_dict[did]['text'] = text
        #did_dict[did]['score'] = scoreDoc.score
        dids.append(did)
        scores.append(scoreDoc.score)

    return dids, scores


def get_lm_doc_snippets(query, searcher, qparser, analyzer, preprocessor, topk=10):
    """
    Fetches the topk document snippets given query, searcher and qparser and
    returns (did, text) pair list
    :param query:
    :param searcher:
    :param qparser:
    :param topk:
    :return:
    """

    dids_text = []

    query = qparser.parse(query)
    scoreDocs = searcher.search(query, topk).scoreDocs

    highlighter = Highlighter(QueryScorer(query))
    highlighter.setTextFragmenter(SimpleFragmenter(100))

    for scoreDoc in scoreDocs:

        doc = searcher.doc(scoreDoc.doc)
        did = doc.get("id")

        text = doc.get("raw")
        token_stream = analyzer.tokenStream("raw", StringReader(text))
        result = highlighter.getBestFragments(token_stream, text, 4, "... ")
        text = get_parsed_text(result)
        text = preprocess_text(preprocessor, [text])
        text = " ".join(text)

        dids_text.append((did, text))

    return dids_text


def get_parsed_text(raw_doc):
    parsed_doc = BeautifulSoup(raw_doc, "lxml")
    doc_text = parsed_doc.get_text().replace('\n', ' ')
    doc_text = ' '.join(doc_text.split())
    return doc_text


def preprocess_text(preprocessor, text):
    text = preprocessor.word_seg_en(text)
    # text = preprocessor.word_stem(text)
    text = preprocessor.word_lower(text)
    text, _ = preprocessor.word_filter(text, preprocessor._word_filter_config, {})
    # text = ' '.join(text[0][0])
    return text[0]


exitFlag = 0
outExitFlag = 0


def write_output_file(out_queue_w, pbar_q):
    while not outExitFlag:
        # qid, dids, scores = out_queue_w.get()
        qid, dids_text = out_queue_w.get()
        tname = multiprocessing.current_process().name
        # print(tname, qid, len(dids))
        # with gzip.open(filtered_queries_dir + 'cw_training_queries_top1000.'+tname+'.txt.gz', 'a') as fout:
        #     for id, doc_id in enumerate(dids):
        #         if doc_id not in docs_dict:
        #             docs_dict[doc_id] = len(docs_dict)
        #         print('{0} {1} {2:.4f}'.format(qid, doc_id, scores[id]), file=fout)
        with open(corpus_base_dir + 'corpus_query_doc_snippets.' + tname + '.v2.txt.gz', 'a') as fout:
            for did, text in dids_text:
                print('{0}+{1}\t{2}'.format(qid, did, text), file=fout)
        pbar_q.put(1)


def process_q_test(q, out_q):
    lucene.initVM()
    lucene.getVMEnv().attachCurrentThread()

    index = DirectoryReader.open(SimpleFSDirectory(Paths.get(robust_index_dir)))
    searcher = IndexSearcher(index)
    searcher.setSimilarity(BM25Similarity())
    analyzer = EnglishAnalyzer()
    qparser = QueryParser("contents", analyzer)
    preprocessor = Preprocess()

    while not exitFlag:
        qid, query = q.get()
        tname = multiprocessing.current_process().name
        # print(tname, qid, query)
        if query == "DONE":
            break

        try:
            # dids, scores = get_lm_matched_docs(query, searcher, qparser, 2000)
            # if len(dids) >= 10:
            #     out_q.put((qid, dids, scores))
            dids_text = get_lm_doc_snippets(query, searcher, qparser, analyzer, preprocessor)
            out_q.put((qid, dids_text))
        except:
            print('%s exception %s, %s' % (tname, qid, query))


def listener(pbar_q):
    pbar = tqdm(total=1000000)  #7588106
    for item in iter(pbar_q.get, None):
        pbar.update()


def check_if_spam(id):
    if id in non_spam_index:
        return False
    return True


def load_spam(file_loc):
    non_spam_doc_ids = set()
    with gzip.open(file_loc,"r") as file:
        for line in file:
            non_spam_doc_ids.add(line.strip())
    return non_spam_doc_ids


if __name__ == '__main__':
    dir = '/home/fernando/aol/AOL-user-ct-collection/'
    corpus_base_dir = '/home/fernando/MatchZoo/data/robust04/triletter_w2v/corpus_v2/'
    filtered_queries_dir = '/home/fernando/aol/filtered_queries/'
    collection_query_file = '/home/fernando/cw09/cw_corpus_q_title.txt'
    robust_index_dir = '/home/singh/indexes/lucene-index.robust04-full.pos.dv'
    cw_index_dir = '/home/singh/cw-idx/lucene-index.cw09b.pos+docvectors'  # '/home/fernando/neuir/cw09b-index'

    # filter_aol_queries(dir, filtered_queries_dir)
    # filtered_queries_for_collection(filtered_queries_dir, collection_query_file)

    # docs_dict_file = '/home/fernando/MatchZoo/data/robust04/docs_dict.txt'
    # docs_dict = {}
    # print("Loading docs dict")
    # with open(docs_dict_file, 'r') as f:
    #     for line in f:
    #         line = line.strip().split()
    #         docs_dict[line[0]] = line[1]
    # print("Completed docs dict loading")

    non_spam_index = None
    # print("loading non-spam index")
    # non_spam_index = load_spam("/home/fernando/cw09/non_spam_doc_ids.txt.gz")
    # print("len of non-spam index %d" % len(non_spam_index))
    # print("done loading index")

    query_sample_file = "/home/fernando/aol/filtered_queries/robust_training_data/robust_q_1m_sample.txt"
    filtered_query_file = "/home/fernando/aol/filtered_queries/queries_robust_filtered.txt.gz"

    qid_list = []
    with open(query_sample_file, "r") as fin:
        for line in fin:
            line = line.strip()
            qid_list.append('Q' + line)
    print("Length of query list %d" % len(qid_list))
    print(qid_list[:10])

    query_map = dict()
    with gzip.open(filtered_query_file, "r") as fin:
        for line in fin:
            line = line.strip().split()
            qid = line[0]
            query = " ".join(line[1:])
            query_map[qid] = query
    print("Length of query map %d" % len(query_map))

    start = datetime.now()
    #lucene.initVM()
    queueLock = Lock()
    manager = multiprocessing.Manager()
    # docs_dict = manager.dict()
    num_threads = 7
    num_out_threads = 5
    work_q = manager.Queue()
    out_queue = manager.Queue()
    pbar_queue = manager.Queue()

    threadList = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
    threads = []
    pbar_worker = multiprocessing.Process(name=threadList[11], target=listener, args=(pbar_queue, ))
    pbar_worker.start()

    for i in range(num_threads):
        worker = multiprocessing.Process(name=threadList[i], target=process_q_test, args=(work_q, out_queue))
        # worker.daemon = True
        worker.start()
        threads.append(worker)

    output_threads = []
    for i in range(num_out_threads):
        output_worker = multiprocessing.Process(name=threadList[i+num_threads], target=write_output_file, args=(out_queue, pbar_queue))
        # output_worker.daemon = True
        output_worker.start()
        output_threads.append(output_worker)

    queueLock.acquire()
    # with gzip.open(filtered_queries_dir + 'queries_cw_not_filtered.txt.gz', 'r') as f:
    #     count = 0
    #     for line in f:
    #         count = count + 1
    #         line = line.strip().split()
    #         qid = line[0]
    #         query = " ".join(line[1:])
    #         work_q.put((qid, query))
    #         # if count == 1000:
    #         #    break
    count = 0
    for qid in tqdm(qid_list):
        count += 1
        work_q.put((qid, query_map[qid]))

    for i in range(num_threads):
        work_q.put((0, "DONE"))
    print('Done putting queries in queue %d' % count)
    queueLock.release()

    for t in threads:
        t.join()

    duration = datetime.now() - start
    print(duration)

    #output_worker.join()
    start = datetime.now()

    while not out_queue.empty():
        pass
    outExitFlag = 1

    duration = datetime.now() - start
    print(duration)
    for t in output_threads:
        t.terminate()

    pbar_queue.put(None)
    pbar_worker.join()
    # print("length of common dict %d" % len(docs_dict))
    print("Exiting Main Thread")
