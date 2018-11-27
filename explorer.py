import pandas as pd
import numpy as np
from knowledgestore import ks

all_article_uris = pd.read_csv("all_article_uris.csv")

def get_events(article_uri):
    """ generates list of event URIs for a given article """
    timecodes = ["#tmx" + str(i) for i in range(7)]
    queries = [
        "SELECT DISTINCT ?event WHERE {?event rdf:type sem:Event . ?event sem:hasAtTime <" + str(article_uri) + str(
            timecode) + ">}" for timecode in timecodes]
    return [result["event"] for query in queries for result in ks.run_sparql_query(query)]


def get_events_with_agent_patient(article_uri):
    """ generates list of event URIs for a given article,where each event has both an agent and a patient """
    timecodes = ["#tmx" + str(i) for i in range(7)]
    queries = [
        "SELECT DISTINCT ?event WHERE {?event rdf:type sem:Event . ?event sem:hasAtTime <" + str(article_uri) + str(
            timecode) +
        "> . ?event propbank:A0 ?_agent . ?event propbank:A1 ?_patient}" for timecode in timecodes]
    return [result["event"] for query in queries for result in ks.run_sparql_query(query)]


def get_mentions_type(article_uri, type):
    """ generates a list of mentions of a certain type for a given article """
    mentions = ks.run_resource_query(article_uri, "ks:hasMention")
    return [m[len(article_uri):] for m in mentions if type in ks.run_mention_query(m, "@type")]


def get_triple_from_event(event_uri):
	query = "SELECT DISTINCT ?agent ?charloc ?patient  WHERE {<" + event_uri + "> propbank:A0 ?agent . <" + event_uri + "> propbank:A1 ?patient . <" + event_uri + "> gaf:denotedBy ?charloc}"
	result = ks.run_sparql_query(query)
	if len(result) == 0:
		return ()
	else:
		agent = (result[0]["agent"].split("/")[-2], result[0]["agent"].split("/")[-1].replace("+", " "))
		patient = (result[0]["patient"].split("/")[-2], result[0]["patient"].split("/")[-1].replace("+", " "))
		charlocs = [(int(r["charloc"].split("=")[-1].split(",")[0]), int(r["charloc"].split("=")[-1].split(",")[1])) for r in result if r["charloc"].split("#")[0] == event_uri.split("#")[0]]
		return (agent, charlocs, patient)


def generate_triples_from_uri(uri):
	events = get_events(uri)
	triples = []
	for event in events:
		trip = get_triple_from_event(event)
		if len(trip) > 0:
			triples.append(trip)
	return triples
