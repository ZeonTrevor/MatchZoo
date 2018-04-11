import lucene

from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import IndexReader, DirectoryReader
from org.apache.lucene.util import BytesRef
from org.apache.lucene.index import MultiFields, PostingsEnum, Term
from org.apache.lucene.search import IndexSearcher, TermQuery

from bs4 import BeautifulSoup

INDEX_BASE_DIR = '/home/singh/indexes/'
INDEX_DIR = 'lucene-index.robust04-full.pos.dv'


def get_lucene_doc(id_field_name, external_doc_id):
    score_docs = searcher.search(TermQuery(Term(id_field_name, external_doc_id)), 1).scoreDocs
    doc_id = score_docs[0].doc
    doc = searcher.doc(doc_id)
    return doc


if __name__ == "__main__":
    lucene.initVM()
    index = DirectoryReader.open(SimpleFSDirectory(Paths.get(INDEX_BASE_DIR + INDEX_DIR)))
    print(index.maxDoc())
    searcher = IndexSearcher(index)
    doc = get_lucene_doc('id', 'FBIS3-10634')
    print(doc.get("raw"))
    parsed_doc = BeautifulSoup(doc.get("raw"), "html5lib")
    text = parsed_doc.get_text().replace('\n', ' ')
    text = ' '.join(text.split())
    print(text)

