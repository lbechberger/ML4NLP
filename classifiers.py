# -*- coding: utf-8 -*-
"""
Example code for training a classifier shown in Session 12.
Created on Mon Jan 21 09:58:26 2019
@author: lbechberger
"""
#Just tiny adjustments to fit code to our dataset
#No imputation needed

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import Imputer
import numpy as np
import pickle

# load the data set (features are positive real numbers)
with open("selected_features.pickle", "rb") as f:
    dataSet = pickle.load(f)

#features = np.array([instance[0] for instance in dataSet])
#targets =  np.array([instance[1] for instance in dataSet])
#print(targets)

data_set = dataSet
X = data_set[0]
y = data_set[1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, shuffle = True)

# set up classifiers        
classifiers = []
classifiers.append(('kNN', KNeighborsClassifier()))
classifiers.append(('NB', GaussianNB()))
classifiers.append(('MaxEnt', LogisticRegression()))
classifiers.append(('DT', DecisionTreeClassifier()))
classifiers.append(('RF', RandomForestClassifier()))
classifiers.append(('SVM', SVC(kernel = 'linear')))
classifiers.append(('MLP', MLPClassifier()))

# train and evaluate
print('\nTRAIN & EVALUATE')
for name, model in classifiers:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    kappa = cohen_kappa_score(y_test, predictions)
    print(name, kappa)
    
# hyperparameter optimization
print('\nGRID SEARCH')
parameter_grid = {'n_neighbors' : np.arange(1,21), 'p': [1, 1.5, 2]}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(),
                           param_grid = parameter_grid,
                           scoring = make_scorer(cohen_kappa_score))
grid_search.fit(X_train, y_train)
print('Best params:', grid_search.best_params_)
predictions = grid_search.predict(X_test)
print('Performance:', cohen_kappa_score(y_test, predictions))

