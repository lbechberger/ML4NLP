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

