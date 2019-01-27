import pandas as pd
import os
from generation_functions import *
from feature_extraction import FeatureExtraction
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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

_, labels = create_dataset(users_db, articles, 100)

# Extract the features
f = FeatureExtraction(articles, categories)
features = f.get_features(users_db)

print("Number of features arrays: {} = {} user * {} articles".format(len(features), len(users_db), len(articles)))

# dimension reduction via filter methods
skb = SelectKBest(score_func=mutual_info_classif, k=3)
skb.fit(features, labels)
print('Feature scores according to mutual information: ', skb.scores_)
filter_transformed = skb.transform(features)
print('After transformation: ', filter_transformed.shape, users_db.shape)
print('Compare: ', features[0], filter_transformed[0])

pickle.dump(filter_transformed, open("./data/filtered_features.pickle", "wb"))


# split dataset into test and training data via k-fold
# kf = KFold(n_splits=10, shuffle=True)
# for train_index, test_index in kf.split(features):
#     X_train = []
#     X_test = []
#     y_train = []
#     y_test = []
#     for i in train_index:
#         X_train.append(features[i])
#         y_train.append(labels[i])
#     for j in test_index:
#         X_test.append(features[j])
#         y_test.append(labels[j])

# print("\nThere are {} categories. In total {} articles resulting in a dataset length of {}"
#       .format(len(categories), len(articles), len(user)))



# for classifier: run 10 times and take average