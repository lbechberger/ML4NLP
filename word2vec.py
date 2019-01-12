import gensim, pickle
from sklearn.metrics.pairwise import cosine_similarity
import knowledgestore.ks as ks
import nltk, string
#import spacy

# Load google news vecs in gensim
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=100000)

def get_summed_word2vec(text):

    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    summed_vector = 0

    for token in tokens:
        if token in model.vocab:
            summed_vector += model[token]

    ms = model.most_similar([summed_vector])

    for x in ms:
        print(x[0],x[1])
    
    return summed_vector
        
def get_features(user):
    
    profile = user[0]
    training = user[1]
    
    """
    for article in profile:
        article_text = ks.run_files_query(article)
        summed_vector = get_summed_word2vec(article_text)

    for article in training[0]:
        article_text = ks.run_files_query(article)
        summed_vector = get_summed_word2vec(article_text)

    for article in training[1]:
        article_text = ks.run_files_query(article)
        summed_vector = get_summed_word2vec(article_text)
    """
        
    article_text = ks.run_files_query(profile[0])
    summed_vector = get_summed_word2vec(article_text)
    
    #print(article)

with open("splitted_dataset.pickle", "rb") as f:
    dataSet = pickle.load(f)
    
user = dataSet[0][0]

