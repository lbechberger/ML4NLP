# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os

# splits dataSet into data for training, testing and validation (plus testing/validation)
# reserves articles to be present only in testing (amount_test_articles) or validation (amount_validation_articles)
    # users that contain articles reserved for testing as well as validation fall into last split testing/validation
def split_dataset(dataSet, amount_test_articles, amount_validation_articles):

    #all articles in dataset
    all_articles = list()

    #creating the list of all articles present in the dataset
    for user in dataSet:
        profile = user[0]
        training = user[1]

        all_articles.extend(profile)
        all_articles.extend(training[0])
        all_articles.extend(training[1])

    #reserve some articles for testing (amount_test_articles) and some for validation (amount_validation_articles)
    test_articles = np.random.choice(all_articles,size = amount_test_articles,replace=False)
    validation_articles = np.random.choice([article for article in all_articles if article not in test_articles],size = amount_validation_articles,replace=False)    

    #lists of users of the later splitted dataset
    training_data = list()
    test_only_data = list()
    validation_only_data = list()

    #a user might contain articles reserved for testing as well as articles reserved for validation, treat those seperately
    validation_and_test_data = list()

    #decide which user belongs to which split of the dataset by checking for the reserved articles
    for user in dataSet:
        profile = user[0]
        training = user[1]

        user_for_test = False
        user_for_validation = False

        for article in profile + training[0] + training[1]:
            if article in test_articles:
                user_for_test = True
            if article in validation_articles:
                user_for_validation = True

        if (not user_for_test) and (not user_for_validation):
            training_data.append(user)
            continue
        if user_for_test and (not user_for_validation):
            test_only_data.append(user)
            continue
        if (not user_for_test) and user_for_validation:
            validation_only_data.append(user)
            continue
        if user_for_test and user_for_validation:
            validation_and_test_data.append(user)
            continue

    
    #encapsulate data
    all_data = list()
    all_data.append(training_data)
    all_data.append(test_only_data)
    all_data.append(validation_only_data)
    all_data.append(validation_and_test_data)

    return all_data

#load dataset
if os.path.isfile('dataset.pickle'):
    with open('dataset.pickle', "rb") as f:
        dataSet = pickle.load(f)

splitted_dataSet = split_dataset(dataSet,10,10)

#print how big the different splits are
for cycle in splitted_dataSet:
    print(len(cycle))

#save splitted dataset
with open('splitted_dataset.pickle','wb') as f:
    pickle.dump(splitted_dataSet,f)
    

