# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:34:38 2018

@author: patri
"""

import knowledgestore.ks as ks
import numpy as np
import csv
import pickle


# creates a list of articles that are not in listOfInterests
def getNegativeExamples(listOfInterests,numberOfExamples,dic):
    #dic = ks.create_category_articles_dictionary()
    
    alreadyChosen = []
    

    all_articles =  [item for sublist in dic.values() for item in sublist]
    
    while len(alreadyChosen) < numberOfExamples:
        article = np.random.choice(all_articles)
        if not article in [art for category in listOfInterests for art in dic[category]]:
            alreadyChosen.append(article)
    return alreadyChosen


# sorts a dictionary by the length of the list of articles of the key category
def count_and_sort(dic):

    # get subcategory-articles matching
    #dic = ks.create_category_articles_dictionary()

    #length of 
    category_lengths=[]

    for key in dic:
        category_lengths.append(len(dic[key]))

    #for i in sorted(cat_lens

    sorted_lengths, sorted_categories = zip(*sorted(zip(category_lengths, dic.keys())))
    for i, element in enumerate(sorted_lengths):
        print(sorted_lengths[i],sorted_categories[i])


# returns a boolean value indicating if the string is a specific date
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

# returns a boolean value indicating whether the string ends with (Wikinewsie)        
def isWikinewsie(keyString):
    lastWord = keyString.split(' ')[-1]
    
    if lastWord == '(Wikinewsie)':
        return True
    else: 
        return False

# returns a boolean value indicating whether the string ends with (WWC2010)     
def isWWC2010(keyString):
    lastWord = keyString.split(' ')[-1]
    
    if lastWord == '(WWC2010)':
        return True
    else: 
        return False

    
# generates a dataset in the format [users:[[userProfile],[[positiveExamples],[negativeExamples]]]]
def generate_dataset(amount_users, subcategories_per_user, articles_per_user_and_category):
    # we only take subcategories, "category" here means subcategory


    # this is done in order to later find twice as many articles: one half for the user profile and one half for the positvie examples for the classifier
    articles_per_user_and_category *= 2

    #
    
    # categories that should not be taken into account:
    categories_to_be_deleted = ['Published','Archived','Original reporting','AutoArchived','Pages with template loops',
                               'Pages using duplicate arguments in template calls','Pages with pull-quotes','Pages with defaulting non-local links','Pages with categorizable local links','Pages using two-parameter languageicon','','Pages with missing-image template calls','Pages with forced foreign links','Pages using three-parameter languageicon',
                               'Reviewed articles','Pages with irredeemable missing-image template calls','Corrected articles','Writing contest 2010','Imported news','Translated news','Featured article','Writing Contests/May 2010','News articles with translated quotes','News articles with telephone numbers']
   
   
   
   
    # get subcategory-articles matching. Create a dictionary
    dic = ks.create_category_articles_dictionary()

    for entry in categories_to_be_deleted:
        del dic[entry]

    
    # get rid of categories that have less articles than articles_per_user_and_category or less than 15 or more than 506
    entries_to_delete=[]
         
    dates_andWikinewsie_list = []    
    for entry in dic:
        if (len(dic[entry]) <= articles_per_user_and_category or len(dic[entry]) <= 15 or len(dic[entry]) > 506): 
            entries_to_delete.append(entry)
    for entry in entries_to_delete:
        del dic[entry]
        
    # get rid of categories that are either a specific date, or a certain author (Wikinewsie) or a certain participant of the Wikinews writing competition (WWC2010)    
    for key in dic.keys():
        if isDate(key) or isWikinewsie(key) or isWWC2010(key):
            dates_andWikinewsie_list.append(key)
    for entry in dates_andWikinewsie_list:
        del (dic[entry])   
        
    
    
    #count_and_sort(dic)
    #all_articles = [item for sublist in dic.values() for item in sublist]
    #print(len(all_articles))

    # computes the weights for each category on the probability distribution (categories with more articles get higher chance)
    normalization_factor = 0
    for entry in dic:
        normalization_factor += len(dic[entry])

    
    length_list=[] # how many articles are in a subcategory
    for entry in dic:
        length_list.append(len(dic[entry])/normalization_factor)


    
    # creating user profiles by drawing random articles from the user's topics of interest
    
    
    users= list()

    # all users
    for user_index in range(amount_users):
        userProfile = []
        userPositives = []
        #userNegatives = []

        # one user
            
                # randomly draw the topics the user is interested in. replace=false means that a user cannot have the same interest multiple times - e.g. like twice as many articles of a certain topic
        sub = np.random.choice(list(dic.keys()),size=subcategories_per_user,replace=False,p= length_list)

                # draw articles from each subcategory
        for topic in sub:
            articles = np.random.choice(dic[topic],size=articles_per_user_and_category,replace=False)
            userProfile.extend(articles[:int(articles_per_user_and_category/2)])
            userPositives.extend(articles[int(articles_per_user_and_category/2):])
                    
                # draw articles the user is not interested in
        userNegatives = getNegativeExamples(sub,   subcategories_per_user * articles_per_user_and_category/2,dic)
        users.append([userProfile,[userPositives,userNegatives]])
        
        

    return users

dataSet = generate_dataset(1000,3,10)

#print(dataSet)
#with open("dataset.csv","w+") as my_csv:
#    csvWriter = csv.writer(my_csv,delimiter=' ')
#    csvWriter.writerows(dataSet)
    
#with open('dataset.pickle','wb') as f:
#    pickle.dump(dataSet,f)
