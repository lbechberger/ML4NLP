# -*- coding: utf-8 -*-
"""
Example code for feature extraction shown in Session 08.

Created on Mon Dec 10 15:48:37 2018

@author: lbechberger
"""

import sys, pickle, time, string

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
print('\nMOST FREQUENT BIGRAMS')
text = ' '.join(articles)
tokens = nltk.word_tokenize(text)
tokens = [token for token in tokens if token not in string.punctuation]
bigrams = nltk.bigrams(tokens)
freq_dist = nltk.FreqDist(bigrams)
frequency_list = []
for bigram, freq in freq_dist.items():
    frequency_list.append([bigram, freq])
frequency_list.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    print(frequency_list[i])

# computing tf-idf
print('\nTF-IDF')
vectorizer = TfidfVectorizer()
tf_idf_vectors = vectorizer.fit_transform(articles).todense()
print(tf_idf_vectors.shape)
print(vectorizer.get_feature_names()[42:45])
print(tf_idf_vectors[:5, 42:45])

tf_idf_similarities = cosine_similarity(tf_idf_vectors)
print(tf_idf_similarities[:5, :5])

# accessing WordNet
print('\nWORDNET')
dog_synsets = wn.synsets('dog')
for syn in dog_synsets:
    words = [str(lemma.name()) for lemma in syn.lemmas()]
    print(syn, words, syn.definition(), syn.hypernyms())

# using word2vec embeddings
# conda install gensim
print('\nWORD2VEC')
embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True).wv
dog_embedding = embeddings['dog']
cat_embedding = embeddings['cat']
canine_embedding = embeddings['canine']
pizza_embedding = embeddings['pizza']
print(cosine_similarity([dog_embedding, cat_embedding, canine_embedding, pizza_embedding]))

man_embedding = embeddings['man']
woman_embedding = embeddings['woman']
king_embedding = embeddings['king']
queen_embedding = embeddings['queen']
summed_embedding = king_embedding - man_embedding + woman_embedding
print(cosine_similarity(queen_embedding.reshape(1, -1), summed_embedding.reshape(1, -1)))

# efficient SPARQL queries
print('\nSPARQL')
mentions = ks.run_resource_query(train[0][0], 'ks:hasMention')
print(len(mentions))
sparql_naive_first = "SELECT ?e WHERE { ?e gaf:denotedBy <"
sparql_naive_second = "> . ?e rdf:type sem:Event }"
sparql_better_first = "SELECT ?e WHERE { VALUES ?m { "
sparql_better_second = " } . ?e gaf:denotedBy ?m . ?e rdf:type sem:Event}"

start_naive = time.time()
naive_results = []
for mention in mentions:
    naive_results += ks.run_sparql_query(sparql_naive_first + mention + sparql_naive_second)
end_naive = time.time()
print(len(naive_results), end_naive - start_naive)

start_better = time.time()
preformatted = list(map(lambda x: '<' + x + '>', mentions))
counter = 0
slice_size = 50
better_results = []
while counter < len(mentions):
    better_results += ks.run_sparql_query(sparql_better_first
                                          + ' '.join(preformatted[counter:counter + slice_size])
                                          + sparql_better_second)
    counter += slice_size
end_better = time.time()
print(len(better_results), end_better - start_better)
