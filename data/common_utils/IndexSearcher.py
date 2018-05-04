from __future__ import print_function
import lucene
import codecs
from datetime import datetime
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.util import BytesRef
from org.apache.lucene.index import MultiFields, PostingsEnum, Term
from org.apache.lucene.search import IndexSearcher, TermQuery, BooleanQuery, BooleanClause
from org.apache.lucene.search.similarities import LMDirichletSimilarity
from bs4 import BeautifulSoup

INDEX_BASE_DIR = '/home/singh/indexes/'
INDEX_DIR = 'lucene-index.robust04-full.pos.dv'

# Read Relation Data
def read_relation(filename, verbose=True):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append( (int(line[0]), line[1], line[2]) )
    if verbose:
        print('[%s]\n\tInstance size: %s' % (filename, len(data)), end='\n')
    return data

# Read Data Dict
def read_data(filename, word_dict = None):
    data = {}
    with open(filename) as f:
        for index, line in enumerate(f):
            if index >= 250:
                break

            line = line.strip().split()
            tid = line[0]
            if word_dict == None:
                data[tid] = list(map(int, line[2:]))
            else:
                data[tid] = []
                for w in line[2:]:
                    wid = int(w)
                    if wid in word_dict:
                        #word_dict[w] = len(word_dict)
                        data[tid].append(word_dict[wid])
                #print(data[tid])
    print('[%s]\n\tData size: %s' % (filename, len(data)), end='\n')
    return data, word_dict


# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    iword_dict = {}
    for line in open(filename):
        line = line.strip().split()
        word_dict[int(line[1])] = line[0]
        iword_dict[line[0]] = int(line[1])
    print('[%s]\n\tWord dict size: %d' % (filename, len(word_dict)), end='\n')
    return word_dict, iword_dict


def get_lm_matched_docs(query, searcher, qparser):
    #did_dict = {}
    dids = []
    scores = []
    query = qparser.parse(query)
    searcher.setSimilarity(LMDirichletSimilarity())
    scoreDocs = searcher.search(query, 2000).scoreDocs
    print("Found %d document(s) that matched query '%s':" % (len(scoreDocs), query))

    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        did = doc.get("id")
        #text = doc.get("raw")
        #did_dict[did] = {}
        #did_dict[did]['text'] = text
        #did_dict[did]['score'] = scoreDoc.score
        dids.append(did)
        scores.append(scoreDoc.score)

    return dids, scores


def store_corpus_docs(data, qrel_docs, searcher, qparser):
    lm_docs = {}

    for qid in data:
        query = ' '.join(data[qid])
        print("qid:%s; query: %s" % (qid, query))
        doc_dict = get_lm_matched_docs(query, searcher, qparser)
        for did in doc_dict:
            if did not in qrel_docs:
                parsed_doc = BeautifulSoup(doc_dict[did], "html5lib")
                text = parsed_doc.get_text().replace('\n', ' ')
                text = ' '.join(text.split())
                lm_docs[did] = text

    print("lm docs not in qrels: %s" % (len(lm_docs)))
    f = codecs.open('/home/fernando/MatchZoo/data/robust04/corpus_n_stem2.txt', 'w', encoding='utf8')
    for did in lm_docs:
        f.write("%s %s\n" % (did, lm_docs[did]))
    f.close()


if __name__ == "__main__":
    lucene.initVM()
    index = DirectoryReader.open(SimpleFSDirectory(Paths.get(INDEX_BASE_DIR + INDEX_DIR)))
    searcher = IndexSearcher(index)
    analyzer = EnglishAnalyzer()
    qparser = QueryParser("contents", analyzer)

    qid_doc_list = {}
    qrel_dict = {}
    qrel_docs = set()

    rel_file = '/home/fernando/MatchZoo/data/robust04/cv_splits/test.5.txt'
    rel = read_relation(filename=rel_file)
    #rel.extend(read_relation(filename='/home/fernando/MatchZoo/data/robust04/relation_train.txt'))
    #rel.extend(read_relation(filename='/home/fernando/MatchZoo/data/robust04/relation_valid.txt'))
    print('Instance size: %s' % (len(rel)), end='\n')
    word_dict, _ = read_word_dict("/home/fernando/MatchZoo/data/robust04/word_dict_new_n_stem_filtered_rob04_embed.txt")

    for label, d1, d2 in rel:
        qrel_dict[(d1, d2)] = label
        qrel_docs.add(d2)

    print('corpus doc size in test rel file: %s' % (len(qrel_docs)), end="\n")

    datapath = '/home/fernando/MatchZoo/data/robust04/corpus_preprocessed_q_n_stem.txt'
    data, _ = read_data(datapath, word_dict)
    qrel_stats = {}
    baseline_f = open('/home/fernando/MatchZoo/data/robust04/cv_splits/predict.test.5.ql.txt','w')
    for label, d1, d2 in rel:
        if d1 not in qid_doc_list:
            qid_doc_list[d1] = []
            query = ' '.join(data[d1])
            print(d1 + " " + query)
            rel, non_rel = 0, 0
            doc_list, scores = get_lm_matched_docs(query, searcher, qparser)

            for id, doc in enumerate(doc_list):

                if (d1, doc) not in qrel_dict:
                    qid_doc_list[d1].append((0, doc))
                    baseline_f.write("%s\t%s\t%s\t%d\t%f\t%s\t%d\n" % (d1, "Q0", doc, id, scores[id], "ql_baseline", 0))
                    non_rel += 1
                else:
                    qid_doc_list[d1].append((qrel_dict[(d1, doc)], doc))
                    baseline_f.write("%s\t%s\t%s\t%d\t%f\t%s\t%d\n" % (d1, "Q0", doc, id, scores[id], "ql_baseline", qrel_dict[(d1, doc)]))
                    rel += 1
            print("rel:%d non-rel:%d" % (rel, non_rel))

        if (label, d2) not in qid_doc_list[d1]:
            # qid_doc_list[d1].append((label, d2))
            if d1 not in qrel_stats:
                qrel_stats[d1] = {}
                qrel_stats[d1]['high_rel'] = 0
                qrel_stats[d1]['rel'] = 0
                qrel_stats[d1]['non_rel'] = 0

            if label == 2:
                qrel_stats[d1]['high_rel'] += 1
            elif label == 1:
                qrel_stats[d1]['rel'] += 1
            else:
                qrel_stats[d1]['non_rel'] += 1

    baseline_f.close()

    for d1 in qrel_stats:
        print("qid:%s high-rel:%d rel:%d non-rel:%d" % (d1, qrel_stats[d1]['high_rel'], qrel_stats[d1]['rel'], qrel_stats[d1]['non_rel']))

    f = open('/home/fernando/MatchZoo/data/robust04/cv_splits/test.5.ql.txt','w')
    for qid in qid_doc_list:
        print("length of docs for qid(%s): %d" % (qid, len(qid_doc_list[qid])))
        for (label, docid) in qid_doc_list[qid]:
            f.write(str(label) + " " + qid + " " + docid + "\n")
    f.close()
