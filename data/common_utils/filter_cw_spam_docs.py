# -*- coding=utf-8 -*-

from __future__ import print_function
import gzip
from tqdm import tqdm

if __name__ == "__main__":
    cw_spam_file = '/home/fernando/cw09/waterloo_spam_scores/clueweb09spam.Fusion'
    cw_filtered_docs_file = '/home/fernando/cw09/non_spam_doc_ids.txt.gz'
    filtered_doc_id_list = []
    with open(cw_spam_file, 'r') as f:
        for line in tqdm(f):
            line = line.strip().split()
            perc_score = int(line[0])
            doc_id = line[1]
            if perc_score >= 70:
                filtered_doc_id_list.append(doc_id)

    print('length of filtered non-spam docs: %d' % len(filtered_doc_id_list))
    with gzip.open(cw_filtered_docs_file, 'w') as fout:
        for doc_id in filtered_doc_id_list:
            print(doc_id, file=fout)