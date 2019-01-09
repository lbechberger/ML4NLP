import pickle, os, csv
import numpy as np
import knowledgestore.ks as ks
import gensim


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
    pickle.dump(articles, open("articles.pickl", "wb"))
    return articles


# def feature_extraction(uris):
#     # if there's already a pickle file in the dir, append features to it; otherwise create neww one
#     if os.path.isfile('./features.pickle'):
#         with open("features.pickle", 'rb') as ppf:
#             features, cat_counter, a_counter = pickle.load(ppf)
#             print("Found pickle file")
#     else:
#         features = []
#         cat_counter = 0
#         a_counter = 0
#         print("Found no pickle file. Creating new list")
# 
#     embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
#                                                                 binary=True).wv
#     for category in uris[cat_counter:]:
#         for a in category[a_counter:]:
#             article_features = []
#             articles_word_list = []
#             if a[4] == 's':  # replace s from https if it exists
#                 a = a[:4] + a[5:]
#             mentions = ks.run_resource_query(a, "ks:hasMention")
#             for m in mentions:
#                 types = ks.run_mention_query(m, "@type")
#                 if "nwr:ObjectMention" not in types:  # or "nwr:EntityMention" in types:
#                     continue
#                 else:
#                     m_title = ks.run_mention_query(m, "nif:anchorOf")
#                     word = ""
#                     counter = 1
#                     # word2vec only accepts single words and not expressions (i.e. Saddam Hussein is an invalid input)
#                     # split string into its words and feed them in the embedding one by one
#                     # need for differentiation between space and end of string because if we add the space to the end of a
#                     # word it becomes an invalid input as well. Same holds for apostrophes: cut word before to include it
#                     # in feature space
#                     for char in m_title[0]:
#                         if char != " " and counter != len(m_title[0]):
#                             word += char
#                             counter += 1
#                             continue
#                         else:
#                             if counter == len(m_title[0]):
#                                 word += char
#                             if len(word) >= 2 and word[-2] == "\'":
#                                 word = word[:-2]
#                             if word not in articles_word_list:
#                                 print("{}: {}".format(m_title, word))
#                                 try:
#                                     articles_word_list.append(word)
#                                     article_features.append(embedding[word])
#                                 except KeyError:
#                                     print("Word \'{}\' not in vocab".format(word))
#                             word = ""
#                         counter += 1
# 
#             features.append(article_features)
#             a_counter += 1
#             print("Article #{} has {} features. In total: {}".format(a_counter,
#                                                                      len(features[cat_counter][a_counter]),
#                                                                      len(features)))
#             # save data to pickle
#             with open('features.pickle', 'wb') as pf:
#                 print("Dumping data ..")
#                 pickle.dump([features, cat_counter, a_counter], pf)
# 
#         cat_counter += 1
#         a_counter = 0
# 
#     return features

def get_w2v_text(text, embedding=[]):
    """
    Get the Google word2vec for each word in an string array
    :param text: string array of words
    :param embedding: embedding matrix. If empty, word2vec from Google will be used
    :return embeddings: array with word2vec values for each word of input text
    """
    forbidden_list = ['and', 'or', 'to', 'from', 'in', 'the', 'for']
    if not embedding:
        print("Found no embedding matrix. Loading Google's word2vec ..")
        embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True).wv
    embeddings = []
    word_list = []
    char_counter = 0
    word = ""
    for char in text[0]:
        if char != " " and char_counter != len(text[0])-1:
            word += char
            char_counter += 1
            continue
        else:
            char_counter += 1
            if char_counter == len(text[0]):
                word += char
            if word[-1] == ',' or word[-1] == '.' or word[-1] == ')':
                word = word[:-1]
            if word[0] == '(':
                word = word[1:]
            if len(word) >= 2 and word[-2] == '\'':
                word = word[:-2]
            if word in forbidden_list:
                print("Skip {}".format(word))
                word = ""
                continue
            if word not in word_list:
                try:
                    print("Include word {}".format(word))
                    embeddings.append(embedding[word])
                    word_list.append(word)
                    word = ""
                except KeyError:
                    print("Unknown word '{}'".format(word))
                    continue
    return embeddings


def get_w2v_string(word_string, embedding=[]):
    """
    Get the Google word2vec for each word in an string array
    :param word_string: single string of words
    :param embedding: embedding matrix. If empty, word2vec from Google will be used
    :return embeddings: array with word2vec values for each word of input text
    """
    forbidden_list = ['and', 'or', 'to', 'from', 'in', 'the', 'for']
    if not embedding:
        print("Found no embedding matrix. Loading Google's word2vec ..")
        embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True).wv
    embeddings = []
    word_list = []
    char_counter = 0
    word = ""
    for char in word_string:
        if char != " " and char_counter != len(word_string)-1:
            word += char
            char_counter += 1
            continue
        else:
            char_counter += 1
            if char_counter == len(word_string):
                word += char
            if word[-1] == ',' or word[-1] == '.' or word[-1] == ')':
                word = word[:-1]
            if word[0] == '(':
                word = word[1:]
            if len(word) >= 2 and word[-2] == '\'':
                word = word[:-2]
            if word in forbidden_list:
                print("Skip {}".format(word))
                word = ""
                continue
            if word not in word_list:
                try:
                    print("Include word {}".format(word))
                    embeddings.append(embedding[word])
                    word_list.append(word)
                    word = ""
                except KeyError:
                    print("Unknown word '{}'".format(word))
                    continue
    return embeddings

