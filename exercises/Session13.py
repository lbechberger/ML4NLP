# -*- coding: utf-8 -*-
"""
Example code for Session 13 (transformers and pipelines).

Created on Mon Jan 28 10:54:59 2019

@author: lbechberger
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
import numpy as np

# make an artificial categorical feature
print('DEALING WITH CATEGORICAL FEATURES')
np.random.seed(42)
X_cat = np.random.choice(['Germany', 'France', 'Russia'], size = (100)) 
probs = {'Germany': 0.9, 'France': 0.6, 'Russia': 0.1}
y_cat = np.array([np.random.choice([1,0], p = [probs[cat], 1 - probs[cat]]) for cat in X_cat])
X_cat = X_cat.reshape(-1, 1)
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat, test_size = 0.30, 
                                                                    random_state = 42, shuffle = True)

print(X_cat_train[:5], y_cat_train[:5])


















## make an artificial continuous data set
#print('\nTRANSFORMERS AND PIPELINES')
#X, y = make_classification(n_samples = 500, n_features = 20, n_informative = 5, 
#                           n_redundant = 5, n_repeated = 2, n_classes = 2, 
#                           n_clusters_per_class = 3, weights = [0.8, 0.2], 
#                           flip_y = 0.05, class_sep = 0.5,
#                           shift = None, scale = None, random_state = 42)
#
## split into training and validation data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, shuffle = True)
#
## naive training
#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)
#predictions = knn.predict(X_test)
#print('naive kNN performance (kappa):', cohen_kappa_score(predictions, y_test))

