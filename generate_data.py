import os
import knowledgestore.ks as ks
import nltk, re, glob
from nltk.sem.relextract import extract_rels, rtuple
import pandas as pd 
import numpy as np

def main():
	dir_setup()
	all_uris = pd.read_csv("all_article_uris.csv")
	for n in range(int(os.environ['SGE_TASK_ID']) - 1, (int(os.environ['SGE_TASK_ID'])+int(os.environ['SGE_TASK_STEPSIZE']))-1):
		data = get_triples(n, all_uris.article[n], int(os.environ['SGE_TASK_ID']))


def main_for_missing_articles():
	"""run if there are jobs killed before they are finished. this will retrieved tripels 
	information from the articles where information is not yet retrieved."""
	missing_articles_id = generate_missing_articles_id()
	all_uris = pd.read_csv("all_article_uris.csv")
	for n in range(int(os.environ['SGE_TASK_ID']) - 1, (int(os.environ['SGE_TASK_ID'])+int(os.environ['SGE_TASK_STEPSIZE']))-1):
		csv_file_name = csv_file_name_for_missing_articles(missing_articles_id[n])
		data = get_triples(missing_articles_id[n], all_uris.article[missing_articles_id[n]], csv_file_name)


def dir_setup():
	"""set up directory"""
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
	
	return mentions


def get_e(mention):
	"""get entities/events given a mention"""
	e = ks.run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <" + mention + ">}")

	return e


def get_mention_string(mention):
	"""get the mention as string"""
	uri = mention.split('#char=')[0]
	start = mention.split('#char=')[1].split(',')[0]
	end = mention.split('#char=')[1].split(',')[1]
	article = ks.run_files_query(uri)
	sentence = article[int(start):int(end)]

	return sentence


def get_word_and_sentence(article, mention):
	"""get the sentence(str) where the mention belongs to"""
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
    else:
        df.to_csv(csv_path, header=False, mode='a')


def get_pos(text):
	"""given text, tokenlize to sentences, words and POS tags"""
	sentences = nltk.sent_tokenize(text)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	return sentences	


def get_triples(article_id, article_uri, csv_file_name):
	"""get triples from article, and save it to csv file.
	For each article,
	1. get all mentions in this article
	2. get all predicates in the mentions
	3. write the sentence where the prerdicate belongs to
	4. find all events with agents, prerdicates and patients
	5. save article id, urri, events, agent, predicate and patient to csv
	"""
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
	

def generate_missing_articles_id():
	"""memory problem might occur in the IKW grid jobs. Thus there might be 
	some articles where triples are not retrieved. This function generate 
	id of these articles. """
	path = './data/raw_csv/' 
	allFiles = glob.glob(path + "/*.csv")
	list_ = []

	for i, file_ in enumerate(allFiles):
	    df = pd.read_csv(file_,index_col=None, header=0)
	    list_.append(df)

	frame = pd.concat(list_, axis = 0, ignore_index = True)
	generated_articles = list(set(frame['id']))
	missing_articles = list(set(list(range(19751))) - set(frame['id']))
	print("number of all articles: 19751")
	print("number of generarted articles: ", len(generated_articles))
	print("number of missing articles: ", len(missing_articles))
	
	return missing_articles


def csv_file_name_for_missing_articles(article_id):
	"""for each missing article, generate csv file name which the triples are wrote to."""
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
	# uncomment this if there is missing article
	# main_for_missing_articles()
