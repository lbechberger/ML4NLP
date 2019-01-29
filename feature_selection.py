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
import matplotlib.pyplot as plt

with open("featurised_dataset3.pickle", "rb") as f:
    dataSet = pickle.load(f)

features = np.array([instance[0] for instance in dataSet])
targets =  np.array([instance[1] for instance in dataSet])
print(targets)
print(len(features),len(targets))

#Up to now only copied example code

# Filter
print('\nFilter')
print('------')
skb = SelectKBest(score_func = mutual_info_classif, k = 18)
skb.fit(features, targets)
print('Feature scores according to mutual information: ', skb.scores_)
filter_transformed = skb.transform(features)
print('After transformation: ', filter_transformed.shape, targets.shape)
print('Compare: ', features[0], filter_transformed[0])

new_set = (filter_transformed, targets)

print(len(dataSet),len(new_set[0][0]))

plt.plot(np.sort(skb.scores_),'ro')
plt.show()

pickle.dump( new_set, open( "selected_features2.pickle", "wb" ) )
