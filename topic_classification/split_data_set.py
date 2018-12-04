# -*- coding: utf-8 -*-
"""
Splits the data set into training and test data.

Created on Mon Dec  3 09:38:10 2018

@author: lbechberger
"""

import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold

parser = argparse.ArgumentParser(description='Data Set Splitter')
parser.add_argument('input_file', help = 'the data set file to use')
parser.add_argument('output_file', help = 'where to store the result')
parser.add_argument('-c', '--cross_validation', action="store_true",
                        help = 'use cross validation instead of train-validation-test split')
parser.add_argument('-f', '--folds', type = int, default = 10,
                        help = 'number of folds for cross validation')
args = parser.parse_args()

with open(args.input_file, "rb") as f:
    data_set = pickle.load(f)

def percentage_of_positive_examples(data_set):
    counter = 0
    for example in data_set:
        if example[1] == 1:
            counter += 1
    return counter / len(data_set)

result = {}

if args.cross_validation:
    # cross valiation
    modified_data_set = []
    groups = []
    
    for category in data_set.keys():
        for example in data_set[category]:
            modified_data_set.append([category] + example)
            groups.append(category)
    modified_data_set = np.array(modified_data_set)
    
    gkf = GroupKFold(n_splits = args.folds)
    all_folds = []
    for train_indices, test_indices in gkf.split(modified_data_set, groups = groups):
        train = modified_data_set[train_indices]
        test = modified_data_set[test_indices]

        print(len(train_indices), len(test_indices), len(test_indices)/len(modified_data_set))
        print(train[0], test[0], '\n')        
        
        all_folds.append(test)

    result['n_folds'] = args.folds
    result['folds'] = all_folds

else:
    # train-validation-test split
    for category in data_set.keys():
        
        # split up
        train, test = train_test_split(data_set[category], test_size = 0.30, random_state = 42, shuffle = True)
        train, validation = train_test_split(train, test_size = 2/7, random_state = 42, shuffle = True)    
        
        p_all = percentage_of_positive_examples(data_set[category])
        p_train = percentage_of_positive_examples(train)
        p_validation = percentage_of_positive_examples(validation)
        p_test = percentage_of_positive_examples(test)
        
        print(category, len(data_set[category]), len(train), len(validation), len(test))
        print(p_all, p_train, p_validation, p_test)
        print(train[0], validation[0], test[0], '\n')  
        
        result[category] = {'train': train, 'validation': validation, 'test': test}
    
with open(args.output_file, "wb") as f:
    pickle.dump(result, f)
