import pandas as pd
from generation_functions import *
from sklearn.model_selection import KFold


# always use same numbers (for performance comparison)
np.random.seed(0)

# load categories, articles and URIs from file
categories, uris, _ = pickle.load(open("user_articles.pickle", "rb"))
# get texts of all articles. Will be dumped in pickle file to avoid unnecessary computations when re-running code
if os.path.isfile('./articles.pickle'):
    articles = pickle.load(open("articles.pickle", "rb"))
else:
    articles = get_articles_from_link(uris)


# create a dataframe with 100 users (rows) who get randomly assigned 0 or 1 for each category (cols)
# correct subcategory labels if superior category is disliked (=0)
users_db = pd.DataFrame(np.random.randint(2, size=(100, len(categories))), columns=categories)
users_db.apply(lambda row: check_subcategories(row), axis=1)


# Embedding of each category
embedding = []
if os.path.isfile('./embed_categories.pickle'):
    print("Found embed_categories.pickle file. Continuing ..")
    embedded_categories = pickle.load(open("embed_categories.pickle", "rb"))
else:
    print("Did not find embed_categories.pickle file. Create embedding of categories ..")
    embedded_categories = []
    embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True).wv
    for c in categories:
        embedded_categories.append(get_w2v_string(c, embedding))
    pickle.dump(embedded_categories, open("embed_categories.pickle", "wb"))


# Embedding of each article
if os.path.isfile('./embed_articles.pickle'):
    embedded_articles, a_counter = pickle.load(open("embed_articles.pickle", "rb"))
else:
    a_counter = 0
    embedded_articles = []
    if not embedding:
        embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                    binary=True).wv
    for a in articles[a_counter:]:
        embed = get_w2v_string(a, embedding)
        mean_embed = [sum(x)/len(x) for x in zip(*embed)]
        embedded_articles.append(mean_embed)
        print("Article: {} /{}".format(a_counter, len(articles)))
        pickle.dump([embedded_articles, a_counter], open("embed_articles.pickle", "wb"))
        a_counter += 1

print("Number of embedded articles: {} with following dimensions: {}".format(len(embedded_articles),
                                                                             len(embedded_articles[0])))


# split dataset into test and training data via k-fold
input_data, labels = create_dataset(users_db, articles, 100)
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(input_data):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in train_index:
        X_train.append(input_data[i])
        y_train.append(labels[i])
    for j in test_index:
        X_test.append(input_data[j])
        y_test.append(labels[j])

print("\nThere are {} categories. In total {} articles resulting in a dataset length of {}"
      .format(len(categories), len(articles), len(input_data)))