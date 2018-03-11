import os
import sys
import six
import io
import array
import numpy as np
import codecs
import pickle
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preprocess import *
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models.keyedvectors import KeyedVectors

def read_dict(infile):
    word_dict = {}
    f = codecs.open(infile, 'r', encoding='utf8')
    
    for line in f:
        r = line.strip().split()
        if len(r) == 2:
            word_dict[r[1]] = r[0]
        else:
            print("line: %s" % line)
            print(r)
    return word_dict

def read_doc(infile):
    dids = list()
    docs = list()
    f = codecs.open(infile, 'r', encoding='utf8')
    for line in tqdm(f):
        r = line.strip().split()
        dids.append(r[0])
        docs.append(r[1:])
        #doc[r[0]] = r[1:]
        #assert len(doc[r[0]]) == int(r[1])
    return dids, docs

def load_word_embedding_gensim(vocab, w2v_file):
    pre_trained = {}
    n_words = len(vocab)
    dim = 300
    embeddings = None
    
if __name__ == '__main__':
    
    w2v_file = sys.argv[1]

    basedir = '../robust04/'
    in_dict_file = basedir + 'word_dict.txt'
    in_corpus_preprocessed_file = basedir + 'corpus_preprocessed_q.txt'
    
    word_dict = {}

    print('load word dict ...')
    word_dict = read_dict(in_dict_file)

    print('Loading docs....')
    dids, docs = read_doc(in_corpus_preprocessed_file)

    vocab = set()
    for doc in docs:
        for wid in doc:
            vocab.add(word_dict[wid]) 
    print('length of vocab in queries %s' % len(vocab))

    model_wv = KeyedVectors.load_word2vec_format(w2v_file)
    count = 0
    
    oov_q = []
    for w in vocab:
        if w in model_wv:
            count += 1
        else:
            oov_q.append(w)

    print('No. of oov query words %s' % len(oov_q))
    pickle.dump(oov_q, open("oov_q.p", "wb" ))
    print('No. of stemmed query words in embeddings %s' % count)
