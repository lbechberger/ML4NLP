import os, glob
import pandas as pd 
import numpy as np


def main():
	data = concatenate_csv(save_csv = True)
	print(data.columns)
	missing_article_ids = generate_missing_articles_id(data)


def generate_missing_articles_id(data):
	"""check number of articles where triples are extracted, where NO triples 
	are extracted, number of triples found, average number of triples per article."""
	
	generated_articles = list(set(data['article_id']))
	missing_articles_id = list(set(list(range(19751))) - set(data['article_id']))
	print("number of all articles: 19751")
	print("number of articles where triples are extracted: ", len(generated_articles))
	print("number of articles where NO triples are extracted: ", len(missing_articles_id))
	print("total number of triples found: ", len(data))
	print("average number of triples per article: ", len(data)/len(generated_articles))
	
	return missing_articles_id

def concatenate_csv(save_csv = True):
	"""concatenate csvs"""
	path = './data/raw_csv/' 
	allFiles = glob.glob(path + "/*.csv")
	list_ = []
	for i, file_ in enumerate(allFiles):
	    df = pd.read_csv(file_,index_col=None, header=0)
	    list_.append(df)

	data = pd.concat(list_, axis = 0, ignore_index = True)
	data = data.rename(columns={'Unnamed: 0': 'event_id', 'id': 'article_id'})

	# sorting by first name 
	data.sort_values(["article_id", "event_id"], ascending=[True, True], inplace=True) 

	# dropping duplicate values 
	data.drop_duplicates(keep='first',inplace=True) 
	if save_csv == True:
		data.to_csv("./data/data_delta.csv", index=False)

	else: pass

	return data

if __name__ == "__main__":
	main()
