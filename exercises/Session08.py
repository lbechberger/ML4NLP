# -*- coding: utf-8 -*-
"""
Example code for feature extraction shown in Session 08.

Created on Mon Dec 10 15:48:37 2018

@author: lbechberger
"""

import sys, pickle, time, string
import sys
sys.path.append(".")
import knowledgestore.ks as ks
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim


print('\nSETUP')
with open("data_set_split.pickle", "rb") as f:
    data_set = pickle.load(f)

train = data_set['Sports']['train']

articles = []
for i in range(10):
    articles.append(ks.run_files_query(train[i][0]))
    
# computing most frequent bigrams
print('\nBIGRAMS')



# computing tf-idf
print('\nTF-IDF')



# accessing WordNet
print('\nWORDNET')



# using word2vec embeddings
# conda install gensim
print('\nWORD2VEC')



# efficient SPARQL queries
print('\nSPARQL')

