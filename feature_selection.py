# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:28:20 2019

@author: patri
"""

from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFromModel
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


with open("feature_selection_three_instances_dataset.pickle", "rb") as f:
    dataSet = pickle.load(f)

features = np.array([instance[0] for instance in dataSet])
targets =  np.array([instance[1] for instance in dataSet])

#Up to now only copied example code

# PCA
print('\nPCA')
print('---')
pca = PCA(random_state = 42)
pca.fit(features)
print('explained percentage of variance: ', pca.explained_variance_ratio_)
print('Most important component: ', pca.components_[0])
pca_transformed = pca.transform(features)
pca_transformed = pca_transformed[:,0:1]
print('After tranformation: ', pca_transformed.shape, targets.shape)
print('Compare: ', features[0], pca_transformed[0])


# Wrapper
print('\nWrapper')
print('-------')
model = LogisticRegression(random_state = 42)
rfe = RFE(model, n_features_to_select = 1)
rfe.fit(features, targets)
print('Features ranked according to RFE: ', rfe.ranking_)
index_of_first = np.where(rfe.ranking_ == 1)[0][0]
index_of_second = np.where(rfe.ranking_ == 2)[0][0]
print('Two most promising features: ', index_of_first, index_of_second)
wrapper_transformed = features[:,[index_of_first,index_of_second]]
print('After tranformation: ', wrapper_transformed.shape, targets.shape)
print('Compare: ', features[0], wrapper_transformed[0])


# Filter
print('\nFilter')
print('------')
skb = SelectKBest(score_func = mutual_info_classif, k = 3)
skb.fit(features, targets)
print('Feature scores according to mutual information: ', skb.scores_)
filter_transformed = skb.transform(features)
print('After transformation: ', filter_transformed.shape, targets.shape)
print('Compare: ', features[0], filter_transformed[0])


# Embedded
print('\nEmbedded')
print('--------')
rf = RandomForestClassifier(n_estimators = 10, random_state = 42)
rf.fit(features, targets)
print('Feature importances of RF classifier: ', rf.feature_importances_)
sfm = SelectFromModel(rf, threshold = 0.1, prefit = True)
embedded_transformed = sfm.transform(features)
print('After transformation: ', embedded_transformed.shape, targets.shape)
print('Compare: ', features[0], embedded_transformed[0])
