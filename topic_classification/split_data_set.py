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

    
    
with open(args.output_file, "wb") as f:
    pickle.dump(result, f)
