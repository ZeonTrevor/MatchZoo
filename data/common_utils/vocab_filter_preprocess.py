
import os
import sys
import numpy as np
import codecs
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preprocess import *
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stopwords

def read_dict(infile):
    word_dict = {}
    f = codecs.open(infile, 'r', encoding='utf8')
    
    for line in tqdm(f):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
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

def load_word_stats(infile):
    words_stats = {}
    f = codecs.open(infile, 'r', encoding='utf8')
    for line in tqdm(f):
        wid, cf, df, idf  = line.split()
        words_stats[wid] = {}
        words_stats[wid]['cf'] = int(cf)
        words_stats[wid]['df'] = int(df)
        words_stats[wid]['idf'] = float(idf)
    return words_stats      

def save_word_stats(word_dict, words_stats, words_stats_fp, sort=False):
    if sort:
        word_dic = sorted(word_dict.items(), key=lambda d:d[1], reverse=False)
        lines = ['%s %d %d %f' % (wid, words_stats[wid]['cf'], words_stats[wid]['df'], words_stats[wid]['idf']) for w, wid in word_dic]
    else:
        lines = ['%s %d %d %f' % (wid, words_stats[wid]['cf'], words_stats[wid]['df'], words_stats[wid]['idf']) for w, wid in word_dict.items()]
    Preprocess.save_lines(words_stats_fp, lines)

if __name__ == '__main__':
    basedir = '../robust04/'
    in_dict_file = basedir + 'word_dict.txt'
    in_corpus_preprocessed_file = basedir + 'corpus_preprocessed.txt'
    in_words_stats_file = basedir + 'word_stats.txt'

    out_corpus_preprocessed_file = basedir + 'corpus_preprocessed_filtered_min10cf.txt'
    out_dict_file = basedir + 'word_dict_filtered_min10cf.txt'
    out_stats_file = basedir + 'word_stats_filtered_min10cf.txt'

    print('Loading word dict...')
    word_dict = read_dict(in_dict_file)
    
    print('Loading docs....')
    dids, docs = read_doc(in_corpus_preprocessed_file)

    print('Loading word stats...')
    words_stats = load_word_stats(in_words_stats_file)
    
    print('Filtering words with frequency less than 10 in the corpus...')
    words_filter_config = {'words_useless': None, 'stop_words': nltk_stopwords.words('english'), 'min_freq': 10}
    docs, words_filtered_ids = Preprocess.word_filter_cf_based(docs, words_filter_config, words_stats)    

    words_dict_filtered = {}
    iwords_dict_filtered = {}
    for wid, word in word_dict.items():
        if wid not in words_filtered_ids:
            words_dict_filtered[word] = wid
            iwords_dict_filtered[wid] = word
    
    print('Remapping wids back to words in documents...')
    docs = [[iwords_dict_filtered[wid] for wid in ws if wid in iwords_dict_filtered] for ws in tqdm(docs)]

    print('Rebuilding word index again from filtered words...')
    docs, new_word_dict = Preprocess.word_index(docs, {'word_dict': None})
    
    new_word_stats = {}
    for w, wid in new_word_dict.items():
        old_wid = words_dict_filtered[w]
        new_word_stats[wid] = {}
        new_word_stats[wid]['cf'] = words_stats[old_wid]['cf']
        new_word_stats[wid]['df'] = words_stats[old_wid]['df']
        new_word_stats[wid]['idf'] = words_stats[old_wid]['idf']

    print('Vocab size after filtering %s' % len(new_word_dict))
    print('Saving filtered vocab...')
    Preprocess.save_dict(out_dict_file, new_word_dict)

    print('Saving filtered word stats...')
    save_word_stats(new_word_dict, new_word_stats, out_stats_file)
 
    print('Saving corpus preprocessed removing min freq words...')
    fout = open(out_corpus_preprocessed_file, 'w')
    for inum, did in enumerate(dids):
        fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
    fout.close()

    print('Completed Filtering Process...')
