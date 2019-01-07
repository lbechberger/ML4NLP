# -*- coding: utf-8 -*-
"""
Dimensionality reduction methods as shown in Session 10.

Created on Mon Jan  7 13:12:49 2019

@author: lbechberger
"""

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# load the data set (features are positive real numbers)
data_set = load_breast_cancer()
features = data_set.data
targets = data_set.target
print('Data set: ', features.shape, targets.shape)
print('Combinatorics of 30 features: ', 2**30)

# PCA
print('\nPCA')
print('---')




# Wrapper
print('\nWrapper')
print('-------')



# Filter
print('\nFilter')
print('------')



# Embedded
print('\nEmbedded')
print('--------')

