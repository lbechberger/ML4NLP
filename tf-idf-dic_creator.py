# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:34:53 2019

@author: patri
"""
import knowledgestore.ks as ks
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

def get_five_highest(uri):
    
    vector = vectorizer.transform([ks.run_files_query(uri)]).toarray()
    print("filesquery",ks.run_files_query(uri))
    if(ks.run_files_query(uri) == ""):
        print("empty")
        return []
    else:
        return five_highest_tf_idf_words(vector)

def five_highest_tf_idf_words(vector):
    vector = np.reshape(vector,-1)
    #print("vector: ",vector)
    argsorted = np.argsort(vector)
    five_highest_indices = argsorted[-5:]
    #print("indices: ",five_highest_indices)
    five_highest_words = []
    for i in five_highest_indices:
        five_highest_words.append(all_words[i])
    return five_highest_words


try:
    vectorizer = pickle.load(open("tf_idf_vectorizer.pickle", "rb"))
except FileNotFoundError:

    all_uris = ks.get_all_resource_uris()


    corpus = []
# append all article's text into one array:

    for uri in all_uris:
        corpus.append(ks.run_files_query(uri))

# compute term frequencies over all words in the corpus
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    all_words = vectorizer.get_feature_names()
#tf_idf_vectors = vectorizer.fit_transform(corpus).toarray()

    
#tf_idic = dict(zip(all_uris,tf_idf_vectors))
# todo : maybe delete numbers.

# compute tf-idf vector for each article

# save it as a dictionary: article_url and tf-idf vector
    pickle.dump( vectorizer, open( "tf_idf_vectorizer.pickle", "wb" ) )
#pickle.dump( tf_idic, open( "tf_idf_dictionary.pickle", "wb" ) )
# save pickle


print(get_five_highest("http://en.wikinews.org/wiki/'Jesus_Camp'_shuts_down"))