# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataset import load_data
from hmm import *

textPath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'train.txt')
datas, tags = load_data(textPath)

################## Bigram HMM start #####################
print('\n************** bigram hmm **************\n')
bihmm = BiHMM(datas, tags) 
print("bigram hmm")
print(bihmm.pred(['小明', '爱', '老鼠', '和', '狗']))
print("bigram hmm with viterbi decode")
print(bihmm.pred(['小明', '爱', '老鼠', '和', '狗'], vb=True))
################## Bigram HMM end   #####################

################## Trigram HMM start #####################
print('\n************* trigram hmm  *************\n')
trihmm = TriHMM(datas, tags) 
print("trigram hmm")
print(trihmm.pred(['小明', '爱', '老鼠', '和', '狗']))
print("trigram hmm with viterbi decode")
print(trihmm.pred(['小明', '爱', '老鼠', '和', '狗'], vb=True))
################## Trigram HMM end   #####################
