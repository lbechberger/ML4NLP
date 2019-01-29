import pandas as pd
import os
from generation_functions import *
from feature_extraction import FeatureExtraction
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score

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

print("Number of feature arrays: {} = {} user * {} articles".format(len(features), len(users_db), len(articles)))

"""
CHOOSE METTHOD HERE: AUSKOMMENTIEREN VON DER GEWUENSCHTEN METHODE; EMBEDDED GIBT BISHER NUR 3 FEATURES ZURUECK 
(KEINE AHNUNG WIESO)
"""
# dimension reduction via filter methods
filtered = f.reduce_dimension(features, labels, 10, "filter")
# filtered_w = f.reduce_dimension(features, labels, 10, "wrapper")
# filtered_e = f.reduce_dimension(features, labels, 10, "embedded")


# split dataset into test and training data via k-fold
# and train with model
# TODO: train classifier 10x and take mean for reliable result
kf = KFold(n_splits=10, shuffle=True)
model = SVC(kernel='linear')
for train_index, test_index in kf.split(filtered):
    X_train = []; y_train = []
    X_test = []; y_test = []
    for i in train_index:
        X_train.append(filtered[i])
        y_train.append(labels[i])
    for j in test_index:
        X_test.append(filtered[j])
        y_test.append(labels[j])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    kappa = cohen_kappa_score(y_test, predictions)
    print(kappa)


# print("\nThere are {} categories. In total {} articles resulting in a dataset length of {}"
#       .format(len(categories), len(articles), len(user)))
