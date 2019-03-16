# -*- coding: utf-8 -*-
# In this program the classifiers are applied to the feature-selected dataset and the classification results are computed with different metrics.

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
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


# returns a string containing the classifiers performance in different metrices
def all_metrics(y, predictions):
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y,predictions)
    f2 = fbeta_score(y,predictions,2)
    kappa = cohen_kappa_score(y, predictions)
    matthew = matthews_corrcoef(y,predictions)

    return "Accuracy: "+str(accuracy)+" F1-score: "+str(f1)+" F2-score: "+str(f2)+" Cohen's_Kappa: "+str(kappa)+" Matthews's_correlation_coefficient: "+str(matthew)

#the metric with which the grid-searches are run
metric = cohen_kappa_score


numbers_of_features = [1,5,6,15]
for i in numbers_of_features:    
    # load the dataset (features are positive real numbers)
    with open("data/reduced_dataset"+str(i)+".pickle", "rb") as f:
        dataset = pickle.load(f)

    
    results_list = []
    
    X_train = np.array([instance[0] for instance in dataset[0]])
    y_train = np.array([instance[1] for instance in dataset[0]])
    X_validation = np.array([instance[0] for instance in dataset[1]])
    y_validation = np.array([instance[1] for instance in dataset[1]])
    X_test = np.array([instance[0] for instance in dataset[2]])
    y_test = np.array([instance[1] for instance in dataset[2]])
    
    X = np.append(X_train,X_validation,axis=0)
    y = np.append(y_train,y_validation,axis=0)

    
    # the test_fold specifies which data in X are used as training and which as validation data during the grid search
    test_fold = np.append(np.array([-1]*len(X_train)),np.zeros(len(X_validation)))
    print(test_fold)
    ps = PredefinedSplit(test_fold=test_fold)
    
    # set up classifiers        
    classifiers = []
    classifiers.append(('kNN', KNeighborsClassifier()))
    #classifiers.append(('NB', GaussianNB()))
    classifiers.append(('MaxEnt', LogisticRegression(solver = 'lbfgs')))
    #classifiers.append(('DT', DecisionTreeClassifier()))
    classifiers.append(('RF', RandomForestClassifier()))
    classifiers.append(('SVM', SVC(kernel = 'linear')))
    classifiers.append(('MLP', MLPClassifier()))
    
    # train and evaluate (without grid search)
    print('\nTRAIN & EVALUATE')
    for name, model in classifiers:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        kappa = cohen_kappa_score(y_test, predictions)
        print(name, all_metrics(y_test,predictions))
        results_list.append((name,"default_parameters"))
        results_list.append(all_metrics(y_test,predictions))
        
    # hyperparameter optimization:
    # Nearest Neighbors
    print('\nGRID SEARCH')
    parameter_grid = {'n_neighbors' : np.arange(1,21), 'p': [1, 1.5, 2]}
    grid_search = GridSearchCV(estimator = KNeighborsClassifier(),
                               param_grid = parameter_grid,
                               scoring = make_scorer(metric),
                               cv=ps)
    grid_search.fit(X, y)
    print('Best params:', grid_search.best_params_)
    predictions = grid_search.predict(X_test)
    print('Performance:', all_metrics(y_test, predictions))
    results_list.append(("kNN",grid_search.best_params_))
    results_list.append(all_metrics(y_test,predictions))
    print("")
    
    # MaxEnt
    parameter_grid = {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    grid_search = GridSearchCV(estimator = LogisticRegression(),
                               param_grid = parameter_grid,
                               scoring = make_scorer(metric),
                               cv=ps)
    grid_search.fit(X, y)
    
    print("")
    print('Best params:', grid_search.best_params_)
    predictions = grid_search.predict(X_test)
    print('Performance:',all_metrics(y_test,predictions))
    results_list.append(("MaxEnt",grid_search.best_params_))
    results_list.append(all_metrics(y_test,predictions))
    
    # random forest
    parameter_grid = {'n_estimators' : np.arange(1,100,5), 'max_depth' : np.arange(110,500,25)}
    parameter_grid['max_depth'] = np.append(parameter_grid['max_depth'],None)
    grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                               param_grid = parameter_grid,
                               scoring = make_scorer(metric),
                               cv=ps)
    grid_search.fit(X, y)
    
    print('Best params:', grid_search.best_params_)
    predictions = grid_search.predict(X_test)
    print('Performance:', all_metrics(y_test,predictions))
    results_list.append(("RF",grid_search.best_params_))
    results_list.append(all_metrics(y_test,predictions))
    
    # save results
    pickle.dump( results_list, open( "data/classifier_results"+str(i)+".pickle", "wb" ) )
