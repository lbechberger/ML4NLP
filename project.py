
import os
import knowledgestore.ks as ks
import nltk, re
from nltk.sem.relextract import extract_rels, rtuple
import pandas as pd 

def run(num_articles):
	all_uris = pd.read_csv("all_article_uris.csv")
	for i, uri in enumerate(all_uris.article[:num_articles]):
		print("article", i)
		agents, preds, patients, events = get_triples(uri)

def get_mentions(article_uri):
	""" generates list of event URIs for a given article """
	prop = 'ks:hasMention'
	mentions = ks.run_resource_query(article_uri, prop)
	print(len(mentions), "mentions")
	# mention_type = [ks.run_mention_query(mention, prop = '@type') for mention in mentions]
	
	return mentions#, mention_type

def get_e(mention):
	"""get entities/events given a mention"""
	e = ks.run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <" + mention + ">}")
	return e

def get_sentence(mention):
	uri = mention.split('#char=')[0]
	start = mention.split('#char=')[1].split(',')[0]
	end = mention.split('#char=')[1].split(',')[1]
	article = ks.run_files_query(uri)
	sentence = article[int(start):int(end)]
	return sentence

def get_triples(article_uri):
	prop = 'ks:hasMention'
	mentions = ks.run_resource_query(article_uri, prop)

	# get all predicates in the mentions
	predicates = [ks.run_mention_query(mention, "nwr:pred") for mention in mentions]

	agents = []
	preds = []
	patients = []
	events = []
	for i, predicate in enumerate(predicates):
		if predicate != []:
			query = "SELECT DISTINCT ?agent ?patient ?event\
				WHERE { \
				?event propbank:A0 ?agent .\
				?event propbank:A1 ?patient .\
				?event gaf:denotedBy <" + mentions[i] + ">}"

			result = ks.run_sparql_query(query)
			if result != []:
				agents.append(result[0]["agent"].split("/")[-1])
				preds.append(predicate[0])
				patients.append(result[0]["patient"].split("/")[-1])
				events.append(result[0]["event"].split("/")[-1])

				print("agent:", result[0]["event"].split("/")[-1])
				print("predicate:", predicate[0])
				print("patient:", result[0]["patient"].split("/")[-1])
				print("event:", result[0]["event"])

	return agents, preds, patients, events
	


if __name__ == "__main__":
	run(100)

