import pickle, re
import gensim
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
import knowledgestore.ks as ks


def create_dataset(database, articles, number_articles_per_categories):
    """
    Creates the data and corresponding label for the whole user database
    :param dataframe database: The database containing all users
    :param list articles: List containing all articles
    :param int number_articles_per_categories: amount of articles in each category
    :return data: array containing userID (array of like/dislike) and articles
    :return labels: array with the like/dislike of each artcile of each user
    """
    labels = []
    data = []
    # user ID = like/dislike of categories
    # data = articles
    # labels = applied likes/dislikes to data
    for user in database.values.tolist():
        for a in articles:
            data.append([user, a])
        # repeats like/dislike n times to assign like/dislike to each article
        labels.append([[like] * number_articles_per_categories for like in user])

    # flatten both lists because we do not need separation of users for classifier
    # TODO improve this
    labels = [item for sublist in labels for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    return data, labels


def check_subcategories(user):
    '''
    If a supercategory is disliked, set subcategories to 0, too.
    Could probably be solved more efficiently, but works for now
    :param dataframe user: The user with random likes/dislikes
    :return dataframe user: The modified user preferences
    '''
    if user[6] == 0:  # science and technology
        user[23] = 0
        user[24] = 0
    if user[7] == 0:  # sports
        user[28] = 0
    if user[12] == 0:  # transport
        user[22] = 0
    if user[10] == 0:  # politics
        user[25] = 0
        user[26] = 0
        user[27] = 0
    if user[17] == 0:  # asia
        user[29] = 0
        user[30] = 0
        user[31] = 0
    if user[18] == 0:  # europe
        user[39] = 0
        user[40] = 0
        user[41] = 0
    if user[19] == 0:  # middle east
        user[32] = 0
        user[33] = 0
    if user[21] == 0:  # oceania
        user[34] = 0
        user[35] = 0
    if user[20] == 0:  # north america
        user[36] = 0
        user[37] = 0
    if user[37] == 0:  # united states
        user[38] = 0
    if user[39] == 0:  # UK
        user[42] = 0
    if user[42] == 0:  # england
        user[43] = 0
    return user


def get_articles_from_link(uris):
    """
    Get the articles' text from their URI's
    :param list uris: The list of all URIs 
    :return list articles: The text of all URIs
    """
    articles = []
    uris = [item for sublist in uris for item in sublist]

    for uri in uris:
        articles.append(ks.run_files_query(uri))

    # save all articles in pickle file
    pickle.dump(articles, open("./data/articles.pickle", "wb"))
    return articles


def get_w2v_string(word_string, embedding):
    """
    Get the Google word2vec for each word in an string array
    :param word_string: single string of words
    :param embedding: embedding matrix
    :return embeddings: array with word2vec values for each word of input text
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    embeddings = []
    # remove periods in acronyms, transform text string into single words and remove duplicates
    word_string = re.sub(r'(?<!\w)([A-Z])\.', r'\1', word_string)
    word_string = set(RegexpTokenizer(r'\w+').tokenize(word_string))

    for word in word_string:
        if word in stop_words:
            continue
        if (len(word) < 4 and not word.isupper()) or word.isdigit():
            continue
        try:
            embeddings.append(embedding[word])
        except KeyError:
            continue
    return embeddings
