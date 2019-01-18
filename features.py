
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:53:46 2019

@author: patri
"""

import knowledgestore.ks as ks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim, pickle, nltk, string



def makeString(listi):
    string = ""
    for i in listi:
        string = string +i+" "
    return string

def get_tf_idf_wordvector(text,numberOfWords,weighted):
    highestWords = highest_tf_idf_words(text, numberOfWords)
    return get_summed_word2vec(makeString(highestWords),weighted)
    

def get_five_highest(article_text):
    
    #article_text = ks.run_files_query(uri)

    #print("filesquery",article_text)
    if(article_text == ""):
        print("empty")
        return []
    else:
        return highest_tf_idf_words(article_text, 5)

def get_tf_idf_scores(text):
    matrix = vectorizer.transform([text])
    matrix = matrix.toarray()
    vector = np.reshape(matrix,-1)

    return vector

def highest_tf_idf_words(text, word_amount):
    vector = vectorizer.transform([text]).toarray()
    vector = np.reshape(vector,-1)

    #print("vector: ",vector)
    argsorted = np.argsort(vector)
    five_highest_indices = argsorted[-word_amount:]
    #print("indices: ",five_highest_indices)
    five_highest_words = []
    for i in five_highest_indices:
        five_highest_words.append(all_words[i])
    return five_highest_words

def initialize_tf_idf():
    global vectorizer, all_words

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
    #tf_idf_vectors = vectorizer.fit_transform(corpus).toarray()

        
    #tf_idic = dict(zip(all_uris,tf_idf_vectors))
    # todo : maybe delete numbers.

    # compute tf-idf vector for each article

    # save it as a dictionary: article_url and tf-idf vector
        pickle.dump( vectorizer, open( "tf_idf_vectorizer.pickle", "wb" ) )
    #pickle.dump( tf_idic, open( "tf_idf_dictionary.pickle", "wb" ) )
    # save pickle

    all_words = vectorizer.get_feature_names()

def initialize_word2vec():
    # Load google news vectors in gensim
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=100000)

def get_summed_word2vec(text, weighted = False):

    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    summed_vector = 0

    if weighted:
        scores = get_tf_idf_scores(text)

    anyword = False #True, if summed_vector contains any word
    for token in tokens:
        if token in model.vocab:
            anyword = True
            if weighted:
                if not token.lower() in all_words:
                    continue
                score = scores[all_words.index(token.lower())]
            else:
                score = 1
            summed_vector += score * model[token]

    #print("Text: ",text,"\nSummed Vec",summed_vector)
    if (anyword == True ):
        ms = model.most_similar([summed_vector])

        #for x in ms:
        #    print(x[0],x[1])
        
        return summed_vector
    else:
        #print(model["for"]*0)
        return model["for"]*0 #Nullvector



def get_distances(articleVector, otherArticlesVectors):
    distancesArray = []
    for otherVector in otherArticlesVectors:
        #print("vector1 : ",articleVector, articleVector.shape)
        #print("vector2 : ",otherVector, otherVector.shape)
        distancesArray.extend(cosine_similarity([articleVector],[otherVector]))
    distancesArray = np.sort(distancesArray)
    return distancesArray[0], distancesArray[-1], np.mean(distancesArray), np.mean(distancesArray[:3])

        
def get_features(user_profile, article_uri):

    features = []
    articles_text = []
    summedVectorsWeighted = []
    summedVectorsUnweighted = []
    fiveHighestVectorsWeighted = []
    fiveHighestVectorsUnweighted = []
    tenHighestVectorsWeighted = []
    tenHighestVectorsUnweighted = []
    tf_idf_scores = []
    lengths = []
    
    article_text = ks.run_files_query(article_uri)

    # five highest tf_idf words in den anderen Artikeln nachschauen
    five_highest_words = get_five_highest(article_text)


    #features for articles in profile
    for profile_article in user_profile:
        articles_text = ks.run_files_query(profile_article)
        summedVectorsWeighted.append(get_summed_word2vec(articles_text,True))
        summedVectorsUnweighted.append(get_summed_word2vec(articles_text,False))
        fiveHighestVectorsWeighted.append(get_tf_idf_wordvector(articles_text,5,True))
        fiveHighestVectorsUnweighted.append(get_tf_idf_wordvector(articles_text,5,False))
        tenHighestVectorsWeighted.append(get_tf_idf_wordvector(articles_text,10,True))
        tenHighestVectorsUnweighted.append(get_tf_idf_wordvector(articles_text,10,False))
        
        # five highest tf_idf words in den anderen Artikeln nachschauen
        scores = get_tf_idf_scores(articles_text)
        score = 0
        for word in five_highest_words:
            if word in all_words:
                score = score + scores[all_words.index(word.lower())]
        tf_idf_scores.append(score)
        
        lengths.append(len(articles_text))
            

    #features for article to be classified

    print("blub1",article_text)
    print(get_summed_word2vec(article_text,True))
    print("\n",summedVectorsWeighted)
    print("\nShape: ",len(summedVectorsWeighted))
    features.append(get_distances(get_summed_word2vec(article_text,True),summedVectorsWeighted))
    print("blub2")
    features.append(get_distances(get_summed_word2vec(article_text,False),summedVectorsUnweighted))
    features.append(get_distances(get_tf_idf_wordvector(article_text,5,True),fiveHighestVectorsWeighted))
    features.append(get_distances(get_tf_idf_wordvector(article_text,5,False),fiveHighestVectorsUnweighted))
    features.append(get_distances(get_tf_idf_wordvector(article_text,10,True),tenHighestVectorsWeighted))
    features.append(get_distances(get_tf_idf_wordvector(article_text,10,False),tenHighestVectorsUnweighted))

    tf_idf_scores = np.sort(tf_idf_scores)
    feature = (tf_idf_scores[0],tf_idf_scores[-1],np.mean(tf_idf_scores),np.mean(tf_idf_scores[-3:]))
    features.append(feature)
    
    distances = [abs(len(article_text)-length) for length in lengths]
    distances = np.sort(distances)
    feature = (distances[0],distances[-1],np.mean(distances),np.mean(distances[:3]))
    features.append(feature)
    
    #features.append(len(article_text)) #length of article

    return features

initialize_tf_idf()
initialize_word2vec()

#print(get_five_highest("http://en.wikinews.org/wiki/'Jesus_Camp'_shuts_down"))

# get_summed_word2vec(ks.run_files_query("http://en.wikinews.org/wiki/'Jesus_Camp'_shuts_down"), True) # <-- works quite well

with open("splitted_dataset.pickle", "rb") as f:
    dataSet = pickle.load(f)
    
user = dataSet[0][0]

features = get_features(user[0],user[1][0][0])
print(features)
