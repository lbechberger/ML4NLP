""" Uses the scripts contained within explorer.py to generate training and test data."""

import pandas as pd
import numpy as np
from knowledgestore import ks
import explorer
from multiprocessing import Pool
import copy
from logging import log
import re

all_article_uris = pd.read_csv("all_article_uris.csv")


def main():
	generate_data_chunks(0, 3)


def generate_data_chunks(start_index, end_index):
	print("Generating classification data for articles {} to {} now.".format(start_index, end_index))
	data_chunk = generate_data_for_articles([all_article_uris.loc[i]["article"] for i in range(start_index, end_index + 1)], gather=True, verbose=True)
	print("Generated data: ")
	print(data_chunk.head())
	print(data_chunk.describe())
	print("----------------------")
	print(data_chunk["classification"].value_counts())
	print("----------------------")
	filename = "cdata_articles_{}_to_{}.csv".format(start_index, end_index)
	data_chunk.to_csv("generated_data/classification_data_chunks/" + filename)
	print("Classification data saved to file {}.".format(filename))


def generate_data_for_articles(articles, gather=False, verbose=False):
	classification_data = pd.DataFrame()
	qap_triples = pd.DataFrame()
	n = 0
	for article_uri in articles:
		if verbose:
			n = n + 1
			print("Processing article {} with URI {} now.".format(n, article_uri))
		triples = explorer.generate_triples_from_article(article_uri)
		article_text = ks.run_files_query(article_uri)
		splits_at = [pos for pos, char in enumerate(article_text) if char == " "]
		words_by_position = [(splits_at[x] + 1, splits_at[x + 1]) for x in range(len(splits_at) - 1)]
		for triple in triples:
			triple_classification = generate_classification_from_triple(triple[0], triple[2], triple[1][0], article_uri, words_by_position)
			classification_data  = classification_data.append(triple_classification, ignore_index=True)
			qap_triples = qap_triples.append(pd.DataFrame({"agent": triple[0], "relation": triple[1][1], "patient": triple[2], "uri": article_uri}, index=[1]), ignore_index=True)
		if gather:
			filename = re.sub(r"[^a-zA-Z0-9]", "_", article_uri.split("/")[-1]) + ".csv"
			qap_triples.to_csv("generated_data/qaps_by_article/" + filename)
	return classification_data


def generate_classification_from_triple(agent, patient, correct_relations, article_uri, words_by_position):
	result_frame = pd.DataFrame()
	for word_by_position in words_by_position:
		classification = word_by_position in correct_relations
		word_row = pd.DataFrame({"agent": agent, "patient": patient, "word_start_char": word_by_position[0], "word_end_char": word_by_position[1], "classification": classification, "article_uri": article_uri}, index=[0])
		result_frame = result_frame.append(word_row, ignore_index=True)
	return result_frame


if __name__ == "__main__":
	main()
