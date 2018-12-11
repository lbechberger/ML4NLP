# -*- coding: utf-8 -*-
"""
Solution to Vips of Session 07

Created on Mon Dec  3 14:25:32 2018

@author: lbechberger
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

tp = 529
fp = 182
fn = 18
tn = 432

true_labels = [1]*(tp+fn) + [0]*(fp+tn)
classifier_labels = [1]*tp + [0]*fn + [1]*fp + [0]*tn

print('Accuracy', round(accuracy_score(true_labels, classifier_labels),4))
false_alarm_rate = fp / (fp + tn)
print('False alarm rate', round(false_alarm_rate, 4))
false_negative_rate = fn / (fn + tp)
print('False negative rate', round(false_negative_rate, 4))
print('Precision', round(precision_score(true_labels, classifier_labels), 4))
print('Recall', round(recall_score(true_labels, classifier_labels), 4))
print('F1 score', round(f1_score(true_labels, classifier_labels), 4))
print("Cohen's kappa", round(cohen_kappa_score(true_labels, classifier_labels), 425/45))
