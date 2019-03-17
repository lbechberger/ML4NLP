# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:44:41 2019
This program writes the results from the classifiers that were saved in a list into a markdown table that can be added into the documentation.
@author: patri
"""

#file = open("classifier_results.txt","r")
#file.close()

import pickle

# this function can be used if you want to know which classifier had the highest f2-score
def get_highest_f2score():
    values = [i.split(" ")[5] for i in results_list if isinstance(i, str)]
    
    #get name of the classifier with the max value
    maxv = max(values)
    indices = [c-1 for c,v in enumerate(results_list) if isinstance(v, str) and v.split(" ")[5] == maxv]
    return [results_list[i] for i in indices]

# functions to print the results in a table. 
def get_table(values):
    table_list = [["Classifier","Accuracy","F1-score","F2-score","Cohen's kappa","Matthews correlation coefficient"]]
    for i in values:
        if (not isinstance(i, str)):
            table_list.append([str(i[0])+" with "+str(i[1])])
        else:
            splitted_string = i.split(" ")
            for j in [1,3,5,7,9]:
                table_list[-1].append(splitted_string[j])
    return table_list

#These functions are taken from https://github.com/lzakharov/csv2md
def get_table_widths(table):
    table_lengths = [[len(cell) for cell in row] for row in table]
    return list(map(max, zip(*table_lengths)))

def table_to_md(table):
    table_widths = get_table_widths(table)

    md_table = ['| ' + ' | '.join([cell.ljust(width) for cell, width in zip(row, table_widths)]) + ' |'
                for row in table]

    md_table.insert(1, '| ' + ' | '.join(['-' * width for width in table_widths]) + ' |')

    return '\n'.join(md_table)

for i in [1,5,6,15]:
    with open("data/classifier_results"+str(i)+".pickle",'rb') as f:
        results_list = pickle.load(f)
    
    print("classifier with highest f2 score: ",get_highest_f2score())
    
    print("number of features: "+str(i))
    print(table_to_md(get_table(results_list)))
    
    
    
                
                
