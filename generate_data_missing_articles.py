import os
import knowledgestore.ks as ks
import nltk, re
from nltk.sem.relextract import extract_rels, rtuple
import pandas as pd 
import numpy as np

def main():
	dir_setup()
	all_uris = pd.read_csv("all_article_uris.csv")
	for n in range(int(os.environ['SGE_TASK_ID']) - 1, (int(os.environ['SGE_TASK_ID'])+int(os.environ['SGE_TASK_STEPSIZE']))-1):
		data = get_triples(n, all_uris.article[n], int(os.environ['SGE_TASK_ID']))


def dir_setup():
	if not os.path.exists('./data'):
		os.mkdir('./data')
	if not os.path.exists('./data/raw_csv'):
		os.mkdir('./data/raw_csv')


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

def get_mention_string(mention):
	uri = mention.split('#char=')[0]
	start = mention.split('#char=')[1].split(',')[0]
	end = mention.split('#char=')[1].split(',')[1]
	article = ks.run_files_query(uri)
	sentence = article[int(start):int(end)]
	return sentence

def get_word_and_sentence(article, mention):
    positions = [int(i) for i in mention.split("#")[1].split("=")[1].split(",")]
    sentences_and_positions = list(zip(nltk.sent_tokenize(article), list(np.cumsum(np.array([len(i) for i in nltk.sent_tokenize(article)])))))
    try:
        correct_sent = list(filter(lambda x: x[1] > positions[0], sentences_and_positions))[0][0]
    except:
        return None, None
    return article[positions[0]:positions[1]], correct_sent

def save_csv(df, csv_file_name):
    csv_path = './data/raw_csv/dataset_delta_' + str(csv_file_name) + '.csv'
    if not os.path.exists(csv_path):
        df.to_csv(csv_path)
        # print('csv saved.')
    else:
        df.to_csv(csv_path, header=False, mode='a')
    # print('csv saved.')


def get_pos(text):
	"""given text, tokenlize to sentences, words and POS tags"""
	sentences = nltk.sent_tokenize(text)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	return sentences	

def get_triples(article_id, article_uri, csv_file_name):
	prop = 'ks:hasMention'
	mentions = ks.run_resource_query(article_uri, prop)

	# get all predicates in the mentions
	predicates = [ks.run_mention_query(mention, "nwr:pred") for mention in mentions]

	data = pd.DataFrame(columns=['id','uri','events','agent','predicate','patients','text'])
	ids = []
	events = []
	texts = []
	# pos = []
	agents = []
	preds = []
	patients = []
	uris = []
	for i, predicate in enumerate(predicates):
		if predicate != []:
			query = "SELECT DISTINCT ?agent ?patient ?event\
				WHERE { \
				?event propbank:A0 ?agent .\
				?event propbank:A1 ?patient .\
				?event gaf:denotedBy <" + mentions[i] + ">}"

			result = ks.run_sparql_query(query)
			
			if result != []:
				sent = get_word_and_sentence(ks.run_files_query(article_uri), mentions[i])[1]
				ids.append(article_id)
				uris.append(article_uri)
				texts.append(sent)
				# pos.append(get_pos(sent))
				agents.append(result[0]["agent"].split("/")[-1])
				preds.append(predicate[0])
				patients.append(result[0]["patient"].split("/")[-1])
				events.append(result[0]["event"].split("/")[-1])

				print("sentence:", sent)
				print("agent:", result[0]["event"].split("/")[-1])
				print("predicate:", predicate[0])
				print("patient:", result[0]["patient"].split("/")[-1])
				print("event:", result[0]["event"])

	data['id'] = ids
	data['uri'] = uris
	data['events'] = events
	data['text'] = texts
	data['agent'] = agents
	data['predicate'] = preds
	data['patients'] = patients
	# data['POS'] = pos
	save_csv(data, csv_file_name)

	return data
	


if __name__ == "__main__":
	main()

