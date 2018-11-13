# -*- coding: utf-8 -*-
"""
Creates the data set for the topic classification.

Creates one binary classification problem for each top level category name.
Stores the result in a single pickle file (dictionary with one list of examples per category name).

Created on Mon Nov 12 14:27:54 2018

@author: lbechberger
"""

import sys
sys.path.append(".")
import knowledgestore.ks as ks

import pickle

result = {}
for category_name in ks.top_level_category_names:
    result[category_name] = []    
    
all_resource_category_mappings = ks.get_all_resource_category_mappings(ks.top_level_category_names)

for resource_uri in ks.get_all_resource_uris():
    for category_name in ks.top_level_category_names:
        line = [resource_uri]
        # if the category_name appears, it's a positive example, otherwise a negative one
        if category_name in all_resource_category_mappings[resource_uri]:
            line.append(1)
        else:
            line.append(0)
        result[category_name].append(line)

for category_name in ks.top_level_category_names:
    print("category: {0}  len(data): {1}".format(category_name, len(result[category_name])))
    print(result[category_name][:10])
    print("")

with open('data_set_raw.pickle', 'wb') as f:
    pickle.dump(result, f)