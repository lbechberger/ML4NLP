import pandas as pd
import os
from generation_functions import *
from feature_extraction import FeatureExtraction
import warnings

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer

warnings.filterwarnings("ignore")
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

# print("Number of feature arrays: {} = {} user * {} articles".format(len(features), len(users_db), len(articles)))

"""
Dimension Reduction
Choose filter method here by uncommenting corresponding line. Might take longer if no pickle file is found
"""
# filtered = f.reduce_dimension(features, labels, 10, "filter")
filtered = f.reduce_dimension(features, labels, 10, "wrapper")
# filtered = f.reduce_dimension(features, labels, 10, "embedded")


# TODO: train classifier 10x and take mean for reliable result (necessary if already using k-split?)
# ('MLP', MLPClassifier()),
# set up classifiers
classifiers = [
    ('kNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('MaxEnt', LogisticRegression()),
    ('DT', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('SVM', LinearSVC()),
]

# split dataset into test and training data via k-fold
# and train with model
kf = KFold(n_splits=10, shuffle=True)
for name, model in classifiers:
    kappa = []
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
        kappa.append(cohen_kappa_score(y_test, predictions))
        print(name, np.mean(kappa))

# # hyperparameter optimization
# print('\nGRID SEARCH')
# parameter_grid = {'n_neighbors': np.arange(1, 21), 'p': [1, 1.5, 2]}
# grid_search = GridSearchCV(estimator=KNeighborsClassifier(),
#                            param_grid=parameter_grid,
#                            scoring=make_scorer(cohen_kappa_score))
# grid_search.fit(X_train, y_train)
# print('Best parameters:', grid_search.best_params_)
# predictions = grid_search.predict(X_test)
# print('Performance:', cohen_kappa_score(y_test, predictions))
