# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:53:46 2019

@author: patri
"""

# this program takes a dataset consisting of user profiles and positive and negative examples as input and returns a set of features.
# The features are explained in more detail in the documentation.

import knowledgestore.ks as ks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim, pickle, nltk, string
import random
import os.path


# converts a list of strings into a large string where the words are seperated by spaces
def makeString(listi):
    string = ""
    for i in listi:
        string = string +i+" "
    return string

# returns a sum of word2vec vectors. It computes the sum for a specified number of words in the text, that have the highest tf-idf values.
def get_tf_idf_wordvector(text,numberOfWords,weighted):
    highestWords = highest_tf_idf_words(text, numberOfWords)
    return get_summed_word2vec(makeString(highestWords),weighted)
    

# returns the five words with the highest tf-idf values for a given article text    
def get_five_highest(article_text):

    if(article_text == ""):
        print("Didn't get article")
        return []
    else:
        return highest_tf_idf_words(article_text, 5)

def get_tf_idf_scores(text):
    matrix = vectorizer.transform([text])
    matrix = matrix.toarray()
    vector = np.reshape(matrix,-1)

    return vector

# returns a specified number of words with the highest tf-idf values for a given article text   
def highest_tf_idf_words(text, word_amount):
    vector = vectorizer.transform([text]).toarray()
    vector = np.reshape(vector,-1)

    argsorted = np.argsort(vector)
    five_highest_indices = argsorted[-word_amount:]
    five_highest_words = []
    for i in five_highest_indices:
        five_highest_words.append(all_words[i])
    return five_highest_words

def initialize_tf_idf():
    global vectorizer, all_words

    # try to load the tf-idf-vectorizer of it already exists
    try:
        vectorizer = pickle.load(open("tf_idf_vectorizer.pickle", "rb"))
    # if it does not exist, make a new one
    except FileNotFoundError:

        # load a list of the uris of all articles
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
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

# returns a sum of the word2vec vectors of all words in an article text. If weighted = true, it is a weighted sum that is 
# weighted acccording to the tf-idf values
def get_summed_word2vec(text, weighted = False):

    # tokenize text
    tokens = nltk.word_tokenize(text)
    # remove punctations
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

    if (anyword == True ):
        #ms = model.most_similar([summed_vector])

        #for x in ms:
        #    print(x[0],x[1])
        
        return summed_vector
    else:
        return model["for"]*0 #Nullvector

def get_entities(article_uri):
    all_entities = set()
    mentions = ks.run_resource_query(article_uri, 'ks:hasMention')
    for mention in mentions:

        #mention_types = ks.run_mention_query(mention, "@type")

        #if "nwr:EntityMention" in mention_types:
            
        entities = ks.run_mention_query(mention, 'ks:refersTo')
        for entity in entities:
            all_entities.add(entity)

    print("ALL ",all_entities)
    return list(all_entities)

# returns cosine similarities between a vector and a set of other vectors. More specifically, it returns the minimum, maximum
# and mean cosine similarity and the mean of the three closest cosine similarities
def get_cosine_similarities(articleVector, otherArticlesVectors):
    distancesArray = []
    for otherVector in otherArticlesVectors:
        #print("vector1 : ",articleVector, articleVector.shape)
        #print("vector2 : ",otherVector, otherVector.shape)
        distancesArray.extend(np.reshape(cosine_similarity([articleVector],[otherVector]),(-1)))
    distancesArray = np.sort(distancesArray)
    return distancesArray[0], distancesArray[-1], np.mean(distancesArray), np.mean(distancesArray[-3:])

def get_features(user_profile, articles_uris, classifications):
    global use_entity_feature

    summedVectorsWeighted = []
    summedVectorsUnweighted = []
    fiveHighestVectorsWeighted = []
    fiveHighestVectorsUnweighted = []
    tenHighestVectorsWeighted = []
    tenHighestVectorsUnweighted = []
    fiveHighestTfIdf = []
    lengths = []
    entities = []


    #features for articles in profile
    for profile_article in user_profile:
        article_text = ks.run_files_query(profile_article)
        summedVectorsWeighted.append(get_summed_word2vec(article_text,True))
        summedVectorsUnweighted.append(get_summed_word2vec(article_text,False))
        fiveHighestVectorsWeighted.append(get_tf_idf_wordvector(article_text,5,True))
        fiveHighestVectorsUnweighted.append(get_tf_idf_wordvector(article_text,5,False))
        tenHighestVectorsWeighted.append(get_tf_idf_wordvector(article_text,10,True))
        tenHighestVectorsUnweighted.append(get_tf_idf_wordvector(article_text,10,False))
        fiveHighestTfIdf.append(get_five_highest(article_text))
        lengths.append(len(article_text))
        if use_entity_feature:
            entities.append(get_entities(profile_article))


    feature_vectors = [] #holds one feature vector for each article defined in parameter articles_uris

    for article_index, article_uri in enumerate(articles_uris):
        print(" Inner loop:",article_index,"/",len(articles_uris))
        feature_vector = []
        article_text = ks.run_files_query(article_uri)
        #features for article to be classified

        feature_vector.extend(get_cosine_similarities(get_summed_word2vec(article_text,True),summedVectorsWeighted))
        feature_vector.extend(get_cosine_similarities(get_summed_word2vec(article_text,False),summedVectorsUnweighted))
        feature_vector.extend(get_cosine_similarities(get_tf_idf_wordvector(article_text,5,True),fiveHighestVectorsWeighted))
        feature_vector.extend(get_cosine_similarities(get_tf_idf_wordvector(article_text,5,False),fiveHighestVectorsUnweighted))
        feature_vector.extend(get_cosine_similarities(get_tf_idf_wordvector(article_text,10,True),tenHighestVectorsWeighted))
        feature_vector.extend(get_cosine_similarities(get_tf_idf_wordvector(article_text,10,False),tenHighestVectorsUnweighted))

        tf_idf_scores = []
        scores = get_tf_idf_scores(article_text)
        for five_highest in fiveHighestTfIdf:
            score = 0
            for word in five_highest:
                if word in all_words:
                    score = score + scores[all_words.index(word.lower())]
            tf_idf_scores.append(score)

        tf_idf_scores = np.sort(tf_idf_scores)
        feature = (tf_idf_scores[0],tf_idf_scores[-1],np.mean(tf_idf_scores),np.mean(tf_idf_scores[-3:]))
        feature_vector.extend(feature)      

        distances = [abs(len(article_text)-length) for length in lengths]
        distances = np.sort(distances)
        feature = (distances[0],distances[-1],np.mean(distances),np.mean(distances[:3]))
        feature_vector.extend(feature)

        if use_entity_feature:
            entity_matching = []
            new_article_entities = get_entities(article_uri)
            for profile_article_entities in entities:
                matching = 0
                for entity in profile_article_entities:
                    if entity in new_article_entities:
                        matching += 1
                entity_matching.append(matching)

            #print(" EM ",entity_matching,"  ",profile_article_entities,"  ",new_article_entities)

            entity_matching = np.sort(entity_matching)
            feature = (entity_matching[0],entity_matching[-1],np.mean(entity_matching),np.mean(entity_matching[-3:]))
            feature_vector.extend(feature)
  
        feature_vectors.append((feature_vector,classifications[article_index]))
        
    return feature_vectors


use_entity_feature = False
n_samples_per_user = 30

initialize_tf_idf()
initialize_word2vec()

# load dataset (splitted into training, validation and test, validation&test category
with open("splitted_dataset.pickle", "rb") as f:
    dataset = pickle.load(f)

# training = dataset[0]
# validation = dataset[1]
# test = dataset[2]

#Continue with feature extraction if it was already started before
if os.path.isfile("featurised_dataset7.pickle"):
    with open("featurised_dataset7.pickle", "rb") as f:
        featurised_dataset = pickle.load(f)
else:
    featurised_dataset = []

n_users = len(dataset[0])+len(dataset[1])+len(dataset[2])
current_user = 0

for purpose_idx, purpose in enumerate(dataset):
    if purpose_idx+1 < len(featurised_dataset): #Continue with feature extraction if it was already started before
        current_user = current_user + int(len(featurised_dataset[purpose_idx])/n_samples_per_user)
        continue
    elif (purpose_idx+1 > len(featurised_dataset)):
        featurised_dataset.append(list())

    for usernumber in range(len(purpose)): #usernumber resets for each purpose, current_user doesn't
        if (usernumber) * n_samples_per_user < len(featurised_dataset[purpose_idx]): #Continue with feature extraction if it was already started before
            current_user = current_user + 1
            continue

        print("\n",str(current_user),"/",str(n_users),"done.","\n")
        current_user += 1

        articles_uris = purpose[usernumber][1][0]+purpose[usernumber][1][1]
        classifications = list(np.ones(len(purpose[usernumber][1][0]), dtype = np.int8)) + list(np.zeros(len(purpose[usernumber][1][1]), dtype = np.int8))
        
        chosen_indices = random.sample(range(0,len(classifications)), n_samples_per_user) #Do not use every article, so variance in user profiles is higher

        chosen_articles_uris = []
        chosen_classifications = []
        for i in chosen_indices:
            chosen_articles_uris.append(articles_uris[i])
            chosen_classifications.append(classifications[i])

        featurised_dataset[-1].extend(get_features(purpose[usernumber][0], chosen_articles_uris, chosen_classifications))

        pickle.dump(featurised_dataset, open( "featurised_dataset7.pickle", "wb" ))


