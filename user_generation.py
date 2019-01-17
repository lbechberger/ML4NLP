import pandas as pd
import os
from generation_functions import *
from feature_extraction import FeatureExtraction
from sklearn.model_selection import KFold

# always use same numbers (for performance comparison)
np.random.seed(0)

# load categories, articles and URIs from file
categories, uris, _ = pickle.load(open("./data/user_articles.pickle", "rb"))
# get texts of all articles. Will be dumped in pickle file to avoid unnecessary computations when re-running code
if os.path.isfile('./data/articles.pickle'):
    articles = pickle.load(open("./data/articles.pickle", "rb"))
else:
    articles = get_articles_from_link(uris)

# create a dataframe with 100 users (rows) who get randomly assigned 0 or 1 for each category (cols)
# correct subcategory labels if superior category is disliked (=0)
users_db = pd.DataFrame(np.random.randint(2, size=(100, len(categories))), columns=categories)
users_db.apply(lambda row: check_subcategories(row), axis=1)


# Extract the features
features = FeatureExtraction(articles, categories)
# calculate the cosine similarities between articles and categories based on user preferences
# TODO: make it work
# a, b, c = features.extract_user_similarities(users_db)
check = features.category_check()
article_lengths = features.get_article_length()
file = open("./data/keywords.txt")
lines = [line.rstrip('\n') for line in file]

# max_similarities, min_similarities, mean_similarities = features.extract_similarities(articles, similarities)
# for val in zip(max_similarities,min_similarities,mean_similarities):
#     print("{}\t\t{}\t\t{}".format(val[0], val[1], val[2]))

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
