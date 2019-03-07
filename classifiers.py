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
from sklearn.metrics import cohen_kappa_score, make_scorer, f1_score, fbeta_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import Imputer
import numpy as np
import pickle


def all_metrics(y, predictions):
    accuracy = accuracy_score(y_validation, predictions)
    f1 = f1_score(y_validation,predictions)
    f2 = fbeta_score(y_validation,predictions,2)
    kappa = cohen_kappa_score(y_validation, predictions)
    matthew = matthews_corrcoef(y_validation,predictions)

    return "Accuracy: "+str(accuracy)+" F1-score: "+str(f1)+" F2-score: "+str(f2)+" Cohen's_Kappa: "+str(kappa)+" Matthews's_correlation_coefficient: "+str(matthew)

# load the data set (features are positive real numbers)
with open("selected_features2.pickle", "rb") as f:
    data_set = pickle.load(f) #TODO spell "dataset" similar everywhere

#features = np.array([instance[0] for instance in dataSet])
#targets =  np.array([instance[1] for instance in dataSet])
#print(targets)

X_train = data_set[0][0]
y_train = data_set[0][1]
X_validation = data_set[1][0]
y_validation = data_set[1][1]

#print(X.shape,y.shape)

#X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.30, random_state = 42, shuffle = True)

# set up classifiers        
classifiers = []
classifiers.append(('kNN', KNeighborsClassifier()))
#classifiers.append(('NB', GaussianNB()))
classifiers.append(('MaxEnt', LogisticRegression(solver = 'lbfgs')))
#classifiers.append(('DT', DecisionTreeClassifier()))
classifiers.append(('RF', RandomForestClassifier()))
classifiers.append(('SVM', SVC(kernel = 'linear')))
classifiers.append(('MLP', MLPClassifier()))

# train and evaluate
print('\nTRAIN & EVALUATE')
for name, model in classifiers:
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    kappa = cohen_kappa_score(y_validation, predictions)
    print(name, kappa)
    
# hyperparameter optimization
print('\nGRID SEARCH')
parameter_grid = {'n_neighbors' : np.arange(1,21), 'p': [1, 1.5, 2]}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(),
                           param_grid = parameter_grid,
                           scoring = make_scorer(cohen_kappa_score))
grid_search.fit(X_train, y_train)
print('Best params:', grid_search.best_params_)
predictions = grid_search.predict(X_validation)
print('Performance:', cohen_kappa_score(y_validation, predictions))
print("")

parameter_grid = {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_search = GridSearchCV(estimator = LogisticRegression(),
                           param_grid = parameter_grid,
                           scoring = make_scorer(cohen_kappa_score))
grid_search.fit(X_train, y_train)

print("")
print('Best params:', grid_search.best_params_)
predictions = grid_search.predict(X_validation)
print('Performance:', cohen_kappa_score(y_validation, predictions))


parameter_grid = {'n_estimators' : np.arange(1,100,5), 'max_depth' : np.arange(110,500,25)}
parameter_grid['max_depth'] = np.append(parameter_grid['max_depth'],None)
grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                           param_grid = parameter_grid,
                           scoring = make_scorer(cohen_kappa_score))
grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
predictions = grid_search.predict(X_validation)
print('Performance:', cohen_kappa_score(y_validation, predictions))

