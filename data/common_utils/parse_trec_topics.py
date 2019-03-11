# -*- coding=utf-8 -*-

from __future__ import print_function

import xml.etree.ElementTree as ET
import os

if __name__ == "__main__":
    cw_topics_dir = '/home/fernando/cw_topics/'
    cw_topics = {}
    for filename in os.listdir(cw_topics_dir):
        print(filename)
        tree = ET.parse(cw_topics_dir + filename)
        root = tree.getroot()
        for topic in root.findall('topic'):
            topic_id = int(topic.get('number'))
            topic_query = topic.find('query').text
            cw_topics[topic_id] = topic_query

    print('length of cw_topics %d' % len(cw_topics))
    sorted_cw_topics = sorted(cw_topics.items(), key=lambda x: x[0], reverse=False)

    with open(cw_topics_dir + 'cw_corpus_q_title.txt','w') as fout:
        for topic_id, topic in sorted_cw_topics:
            print(topic_id, topic, file=fout)