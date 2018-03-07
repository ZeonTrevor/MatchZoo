#!/bin/bash

w2v_FILE = ../../../drrm/wordembedding/glove.840B.300d.txt
word_dict_FILE = ../robust04/word_dict_filtered_min10cf.txt
output_embed_FILE = ../robust04/embed_glove_rob04_af_d300
additional_w2v_FILE = ../../../drrm/wordembedding/rob04.d300.txt

python gen_w2v.py $w2v_FILE $word_dict_FILE $output_embed_FILE $additional_w2v_FILE
