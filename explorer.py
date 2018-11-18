import pandas as pd
import numpy as np
from knowledgestore import ks

all_article_uris = pd.read_csv("all_article_uris.csv")

def main():
	# select random article from all articles
	article_uri = all_article_uris.loc[np.random.randint(low=0, high=len(all_article_uris.index))]["article"]
	print(article_uri)
	relation_mentions = get_mentions_type(article_uri, "nwr:RelationMention")
	events = get_events(article_uri)
	print(relation_mentions)
	print(events)


def get_events(article_uri):
	""" generates list of event URIs for a given article """
	timecodes = ["#tmx" + str(i) for i in range(7)]
	queries = ["SELECT DISTINCT ?event WHERE {?event rdf:type sem:Event . ?event sem:hasAtTime <" + str(article_uri) + str(timecode) + ">}" for timecode in timecodes]
	return [result["event"] for query in queries for result in ks.run_sparql_query(query)]


def get_mentions_type(article_uri, type):
	""" generates a list of mentions of a certain type for a given article """
	mentions = ks.run_resource_query(article_uri, "ks:hasMention")
	return [m[len(article_uri):] for m in mentions if type in ks.run_mention_query(m, "@type")]


if __name__ == "__main__":
	main()
