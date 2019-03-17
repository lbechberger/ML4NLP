# -*- coding: utf-8 -*-
# splits dataset into data for training, testing and validation

import pickle
import numpy as np
import os


# reserves articles to be present only in testing (amount_test_articles) or validation (amount_validation_articles)
    # users that contain articles reserved for testing as well as validation fall into last split testing/validation
def split_dataset(dataset, n_validation_articles, n_test_articles):

    #all articles in dataset
    all_articles = list()

    #creating the list of all articles present in the dataset
    for user in dataset:
        profile = user[0]
        training = user[1]

        all_articles.extend(profile)
        all_articles.extend(training[0])
        all_articles.extend(training[1])

    #reserve some articles for testing (amount_test_articles) and some for validation (amount_validation_articles)
    test_articles = np.random.choice(all_articles,size = n_test_articles,replace=False)
    validation_articles = np.random.choice([article for article in all_articles if article not in test_articles],size = n_validation_articles,replace=False)    

    #lists of users of the later splitted dataset
    training_data = list()
    validation_only_data = list()
    test_only_data = list()

    #decide which user belongs to which split of the dataset by checking for the reserved articles
    for user in dataset:
        profile = user[0]
        training = user[1]

        user_for_test = False
        user_for_validation = False

        for article in profile + training[0] + training[1]:
            if article in test_articles:
                user_for_test = True
            if article in validation_articles:
                user_for_validation = True

        if (not user_for_validation) and (not user_for_test):
            training_data.append(user)
            continue
        if (not user_for_test) and user_for_validation:
            validation_only_data.append(user)
            continue
        if user_for_test and (not user_for_validation):
            test_only_data.append(user)
            continue

        #a user might contain articles reserved for testing as well as articles reserved for validation, discard that user from dataset
    
    #encapsulate data
    all_data = list()
    all_data.append(training_data)
    all_data.append(validation_only_data)
    all_data.append(test_only_data)

    return all_data

#load dataset
with open('data/dataset.pickle', "rb") as f:
    dataset = pickle.load(f)

splitted_dataset = split_dataset(dataset,10,5)

#print how big the different splits are
print("Users for Training:",len(splitted_dataset[0]))
print("Users for Validation:",len(splitted_dataset[1]))
print("Users for Test:",len(splitted_dataset[2]))

#save splitted dataset
with open('data/splitted_dataset.pickle','wb') as f:
    pickle.dump(splitted_dataset,f)
