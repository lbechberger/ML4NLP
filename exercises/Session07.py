# -*- coding: utf-8 -*-
"""
Example for computing different evaluation metrics as shown in Session 07.

Created on Mon Dec  3 11:53:50 2018

@author: lbechberger
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

data_set_1 = [1]*500 + [0]*500
data_set_2 = [1]*100 + [0]*900

data_sets = {'Data Set 1': data_set_1, 'Data Set 2': data_set_2}

always_true = DummyClassifier(strategy='constant', constant=1)
always_false = DummyClassifier(strategy='constant', constant=0)
fifty_fifty = DummyClassifier(strategy='uniform')
frequency = DummyClassifier(strategy='stratified')

classifiers = {'at': always_true, 'af': always_false, '50-50': fifty_fifty, 'freq': frequency}

for data_set_name, data_set in data_sets.items():
    print(data_set_name)
    for classifier_name, classifier in classifiers.items():
        dummy_data = [[None]]*1000
        classifier.fit(dummy_data, data_set)
        predictions = classifier.predict(dummy_data)
        
        accuracy = accuracy_score(data_set, predictions)

        tn, fp, fn, tp = confusion_matrix(data_set, predictions).ravel()
        false_alarm_rate = fp / (fp + tn)
        false_negative_rate = fn / (fn + tp)

        precision = precision_score(data_set, predictions)
        recall = recall_score(data_set, predictions)
        f_score = f1_score(data_set, predictions)

        kappa = cohen_kappa_score(data_set, predictions)

        print("\t {0}: Acc {1}, FAR {2}, FNR {3}, Prec {4}, Rec {5}, F {6}, kappa {7}".format(classifier_name, accuracy, false_alarm_rate, false_negative_rate, precision, recall, f_score, kappa))        
        


        
        