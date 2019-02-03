import pandas as pd
import os
import math
from generation_functions import *
from feature_extraction import FeatureExtraction
import warnings

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
filter_method = 'filter'
# filter_method = 'wrapper'

filtered = f.reduce_dimension(features, labels, 10, filter_method)
# filtered = f.reduce_dimension(features, labels, 10, "wrapper")
# print(np.any(np.isnan(filtered))) # check for NaN values
# TODO: train classifier 10x and take mean for reliable result (necessary if already using k-split?)
#     ('MLP', MLPClassifier()),
#     ('NB', GaussianNB()),
#     ('MaxEnt', LogisticRegression()),
#     ('DT', DecisionTreeClassifier()),
#     ('SVM', LinearSVC()),

# set up classifiers
classifiers = [
    ('kNN', KNeighborsClassifier(algorithm='auto')),
    ('RF', RandomForestClassifier()),
]

# split dataset into test and training data via k-fold
# and train with model
kf = KFold(n_splits=10, shuffle=True)
parameter_grid_knn = {'n_neighbors': np.arange(1, 21), 'weights': ['uniform', 'distance'], 'p': [1, 1.5, 2]}
parameter_grid_rf = {'n_estimators': np.arange(1, 100), 'bootstrap': ['True, False']}
print("Dimension Reduction based on {} Methods. Optimisation via Grid Search".format(filter_method))
for name, model in classifiers:
    kappa_before = []; kappa_after = []
    X_train, X_test, y_train, y_test = train_test_split(filtered, labels)
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
        kappa_before.append(cohen_kappa_score(y_test, predictions))

        # hyperparameter optimization
        # depending on model, use different parameter grid
        parameter_grid = parameter_grid_knn if name == 'kNN' else parameter_grid_rf
        grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring=make_scorer(cohen_kappa_score))
        grid_search.fit(X_train, y_train)
        print('Best parameters:', grid_search.best_params_)
        predictions = grid_search.predict(X_test)
        kappa_after.append(cohen_kappa_score(y_test, predictions))
    print(name, "before grid search:", np.mean(kappa_before))
    print(name, "after grid search:", np.mean(kappa_after))
