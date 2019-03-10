
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:28:20 2019
@author: patri

Reduces dimensionality of dataset by applying filter (mutual information) and/or embedded (random forest) method.
"""

from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFromModel
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys

with open("featurised_dataset7.pickle", "rb") as f:
    dataset = pickle.load(f)

training = dataset[0]
validation = dataset[1]
test = dataset[2]

features = np.array([instance[0] for instance in training])
targets =  np.array([instance[1] for instance in training])

selected_features = list()

use_filter = True #Use filter method for selection (mutual information)
use_embedded = True #Use embedded method for selection (random forest)
n_features_filter = 5 #Amount of features to select by filter method
n_features_embedded = 5 #Amount of features to select by embedded method

if not (use_filter or use_embedded):
    print("No selection method selected. Exiting.")
    sys.exit()


highest_features_filter = []
highest_features_embedded = []

if use_filter: #Filter method selection
    print('\nFilter')
    print('------')
    skb = SelectKBest(score_func = mutual_info_classif, k = 'all')
    skb.fit(features, targets)
    print('\nFeature scores according to mutual information: ', skb.scores_)

    #Copy indices of n_features_filter best features in highest_features_filter
    for i in range(1,n_features_filter+1):
        highest_features_filter.append(list(skb.scores_).index(np.sort(skb.scores_)[-i]) )
    print("\nHighest feature indices according to filter:",highest_features_filter)

if use_embedded: # Embedded method selection
    print('\n\nEmbedded')
    print('------')

    rf = RandomForestClassifier(n_estimators = 10, random_state = 42)
    rf.fit(features, targets)
    print('\nFeature importances of RF classifier: ', rf.feature_importances_)
    sfm = SelectFromModel(rf, threshold = 0, prefit = True)

    #Copy indices of n_features_embedded best features in highest_features_embedded
    for i in range(1,n_features_embedded+1):
        highest_features_embedded.append(list(rf.feature_importances_).index(np.sort(rf.feature_importances_)[-i]) )
    print("\nHighest feature indices according to embedded:",highest_features_embedded)

#Combine selected features from the two methods
selected_features = set(highest_features_embedded + highest_features_filter)
print("\nFeatures used:",selected_features)

#Reduce dimensionality of dataset by using only the selected features
reduced_dataset = []
for purpose in dataset:
    reduced_purpose = []
    for sample in purpose:
        reduced_sample = []
        for feature_index in selected_features:
            reduced_sample.append(sample[0][feature_index]) # index 0: feature vector
        reduced_purpose.append([reduced_sample,sample[1]]) # index 1: classification
    reduced_dataset.append(reduced_purpose)

#Plot scores
if (use_filter):
    plt.plot(np.sort(skb.scores_),'bo',label = "Filter method")
if (use_embedded):
    plt.plot(np.sort(rf.feature_importances_),'ro',label = "Embedded method")
plt.legend()
plt.ylabel('Importance score')
plt.xlabel('Feature (sorted by importance score)')
plt.show()

#Save reduced dataset
pickle.dump(reduced_dataset, open("reduced_dataset.pickle", "wb" ) )

