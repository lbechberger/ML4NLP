import knowledgestore.ks as ks
import numpy as np

def count_and_sort():

    # get subcategory-articles matching
    dic = ks.create_category_articles_dictionary()

    #length of 
    category_lengths=[]

    for key in dic:
        category_lengths.append(len(dic[key]))

    #for i in sorted(cat_lens

    sorted_lengths, sorted_categories = zip(*sorted(zip(category_lengths, dic.keys())))
    for i, element in enumerate(sorted_lengths):
        print(sorted_lengths[i],sorted_categories[i])

count_and_sort()
