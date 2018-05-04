# USAGE
# python simple_request.py
import requests
import lucene
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.index import Term
from org.apache.lucene.search import IndexSearcher, TermQuery
from bs4 import BeautifulSoup


def get_lucene_doc(id_field_name, external_doc_id):
    score_docs = searcher.search(TermQuery(Term(id_field_name, external_doc_id)), 1).scoreDocs
    doc_id = score_docs[0].doc
    doc = searcher.doc(doc_id)
    return doc


INDEX_BASE_DIR = '/home/singh/indexes/'
INDEX_DIR = 'lucene-index.robust04-full.pos.dv'

DRMM_REST_API_URL = "http://localhost:5000/score"
query = "international organized crime"  # Robust Track #301
non_rel_doc_id = 'FT921-3124'
rel_doc_id = 'FBIS3-10082'

lucene.initVM()
index = DirectoryReader.open(SimpleFSDirectory(Paths.get(INDEX_BASE_DIR + INDEX_DIR)))
searcher = IndexSearcher(index)

doc = get_lucene_doc('id', rel_doc_id)
parsed_doc = BeautifulSoup(doc.get("raw"), "html5lib")
doc_cont = parsed_doc.get_text().replace('\n', ' ')
doc_cont = ' '.join(doc_cont.split())
print(doc.get("raw"))
payload = {"query": query, "doc": doc_cont}

r = requests.post(DRMM_REST_API_URL, json=payload).json()
print(float(r))
