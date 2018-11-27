# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:34:38 2018

@author: patri
"""

import knowledgestore.ks as ks
import numpy as np
import csv
import pickle


def isDate(dateString):
    months = ['January','February','March','April','May','June','July','August','September','October','November','December']    
    splitted = dateString.split(' ',1)
    firstWord = splitted[0]
        
    if (firstWord in months):
        secondWord = splitted[1].split(' ',1)[0]
        try:
            secondWord = secondWord.replace(',', '')
            int(secondWord)
            return True
        except ValueError:
            return False
        
def isWikinewsie(keyString):
    lastWord = keyString.split(' ')[-1]
    
    if lastWord == '(Wikinewsie)':
        return True
    else: 
        return False
    
def articles0(keyString):
    if keyString.split(' ')[-1] == 'articles':
        return True
    else:
        return False
    

def generate_dataset(amount_users, subcategories_per_user, articles_per_user_and_category):
    # we only take subcategories, "category" here means subcategory

    # categories that should not be taken into account:
    categories_to_be_deleted = ['Published','Archived','Original reporting','AutoArchived','Pages with template loops',
                               'Pages using duplicate arguments in template calls','Pages with pull-quotes','Pages with defaulting non-local links','Pages with categorizable local links','Pages using two-parameter languageicon','','Pages with missing-image template calls','Pages with forced foreign links','Pages using three-parameter languageicon',
                               'Reviewed articles','Pages with irredeemable missing-image template calls','Corrected articles','Writing contest 2010','Imported news','Translated news','Featured article']
   # maybe = ['News articles with telephone numbers',] # Konrektes Datum? 
   
   
   
    # get subcategory-articles matching
    dic = ks.create_category_articles_dictionary()

    for entry in categories_to_be_deleted:
        del dic[entry]

    
    # get rid of categories that have less articles than articles_per_user_and_category+1 and more than (500?)
    entries_to_delete=[]
         
    dates_andWikinewsie_list = []    
    for entry in dic:
        if (len(dic[entry]) <= articles_per_user_and_category or len(dic[entry]) > 506): 
            entries_to_delete.append(entry)
    for entry in entries_to_delete:
        del dic[entry]
    for key in dic.keys():
        if isDate(key) or isWikinewsie(key):
            dates_andWikinewsie_list.append(key)
    for entry in dates_andWikinewsie_list:
        del (dic[entry])   
        
    
    
    
    for entry in dic.keys():
        if articles0(entry):
            print(entry)
# todo : remove categories like "published" , "AutoArchived", "Pages with defaulting non-local links", and maybe specific dates


    # computes the weights for each category on the probability distribution (categories with more articles get higher chance)
    normalization_factor = 0
    for entry in dic:
        normalization_factor += len(dic[entry])

    
    length_list=[] # how many articles are in a subcategory
    for entry in dic:
        length_list.append(len(dic[entry])/normalization_factor)


    
    # creating user profiles by drawing random articles from the user's topics of interest
    users=list()

    # all users
    for user_index in range(amount_users):
        user_articles = list()

        # one user
        for sub_index in range(subcategories_per_user):
            
                # randomly draw the topics the user is interested in. replace=false means that a user cannot have the same interest multiple times - e.g. like twice as many articles of a certain topic
                sub = np.random.choice(list(dic.keys()),size=subcategories_per_user,replace=False,p= length_list)

                # draw articles from each subcategory
                for topic in sub:
                    articles = np.random.choice(dic[topic],size=articles_per_user_and_category,replace=False)
                    user_articles.extend(articles)


        users.append(user_articles)

    return users

#print(generate_dataset(100,3,10))
dataSet = generate_dataset(100,3,10)

with open("dataset.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=' ')
    csvWriter.writerows(dataSet)
    
wiht open('dataset.pickle','wb') as f:
    pickle.dump(dataset,f)