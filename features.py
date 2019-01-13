import knowledgestore.ks as ks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim, pickle, nltk, string

def get_five_highest(article_text):
    
    #article_text = ks.run_files_query(uri)

    #print("filesquery",article_text)
    if(article_text == ""):
        print("empty")
        return []
    else:
        return highest_tf_idf_words(article_text, 5)

def tf_idf_scores(text):
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
        scores = tf_idf_scores(text)

    for token in tokens:
        if token in model.vocab:
            if weighted:
                if not token.lower() in all_words:
                    continue
                score = scores[all_words.index(token.lower())]
            else:
                score = 1
            summed_vector += score * model[token]

    ms = model.most_similar([summed_vector])

    for x in ms:
        print(x[0],x[1])
    
    return summed_vector
        
def get_features(user_profile, article_uri):

    features = []
    articles_text = []

    #features for articles in profile
    for profile_article in user_profile:
        articles_text.append(ks.run_files_query(profile_article))

    #features for article to be classified
    articles_text.append(ks.run_files_query(article_uri))

    for article_text in articles_text:
        features.append(len(article_text)) #length of article
        features.append(get_five_highest(article_text)) #5 words with highest tf-idf-score
        features.append(get_summed_word2vec(article_text, False)) #summed word vector
        features.append(get_summed_word2vec(article_text, True)) #summed word vector with words weighted according to their tf-idf score

    ## TODO: relate features, so features aren't single vectors, but distances between vectors, maybe minimum distance

    return features

initialize_tf_idf()
initialize_word2vec()

#print(get_five_highest("http://en.wikinews.org/wiki/'Jesus_Camp'_shuts_down"))

# get_summed_word2vec(ks.run_files_query("http://en.wikinews.org/wiki/'Jesus_Camp'_shuts_down"), True) # <-- works quite well

with open("splitted_dataset.pickle", "rb") as f:
    dataSet = pickle.load(f)
    
user = dataSet[0][0]

features = get_features(user[0],user[1][0][0])


