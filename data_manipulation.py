import os, glob
import pandas as pd 
import numpy as np


def main():
	missing_article_ids = generate_missing_articles_id()

def generate_missing_articles_id():
	path = './data/raw_csv/' 
	allFiles = glob.glob(path + "/*.csv")

	list_ = []

	for i, file_ in enumerate(allFiles):
	    df = pd.read_csv(file_,index_col=None, header=0)
	    list_.append(df)

	data = pd.concat(list_, axis = 0, ignore_index = True)
	# sorting by first name 
	data.sort_values("id", inplace=True) 

	# dropping duplicate values 
	data.drop_duplicates(keep='first',inplace=True) 
  
	generated_articles = list(set(data['id']))
	missing_articles = list(set(list(range(19751))) - set(data['id']))
	print("number of all articles: 19751")
	print("number of generarted articles: ", len(generated_articles))
	print("number of missing articles: ", len(missing_articles))
	print("number of triples found: ", len(data))
	print("average number of triples per article: ", len(data)/len(generated_articles))
	return missing_articles

def csv_file_name_for_missing_articles(article_id):
	len_digit = len(str(article_id))
	if len_digit < 4:
		file_name = "dataset_delta_1.csv"
		file_id = str(1)
	else: 
		beginning_digit = str(article_id)[:-3]
		file_name = "dataset_delta_" + beginning_digit + "001.csv"
		file_id = beginning_digit + "001"
	return file_id

if __name__ == "__main__":
	main()
