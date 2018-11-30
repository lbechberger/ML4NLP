from knowledgestore import ks


def get_events(article_uri):
	""" generates a list of event URIs for a given article URI"""
	timecodes = ["#tmx" + str(i) for i in range(7)]
	queries = [
		"SELECT DISTINCT ?event WHERE {?event rdf:type sem:Event . ?event sem:hasAtTime <" + str(article_uri) + str(
			timecode) + ">}" for timecode in timecodes]
	return [result["event"] for query in queries for result in ks.run_sparql_query(query)]


def get_triple_from_event(event_uri):
	""" generates a triples from a given event URI """
	query = "SELECT DISTINCT ?agent ?charloc ?patient  WHERE {<" + event_uri + "> propbank:A0 ?agent . <" + event_uri +\
			"> propbank:A1 ?patient . <" + event_uri + "> gaf:denotedBy ?charloc}"
	result = ks.run_sparql_query(query)
	if len(result) == 0:
		return ()
	else:
		agent = (result[0]["agent"].split("/")[-2], result[0]["agent"].split("/")[-1].replace("+", " "))
		patient = (result[0]["patient"].split("/")[-2], result[0]["patient"].split("/")[-1].replace("+", " "))
		charlocs = [(int(r["charloc"].split("=")[-1].split(",")[0]), int(r["charloc"].split("=")[-1].split(",")[1])) for
					r in result if r["charloc"].split("#")[0] == event_uri.split("#")[0]]
		return (agent, charlocs, patient)


def generate_triples_from_article(article_uri):
	""" generates a list of triples from a given article URI """
	events = get_events(article_uri)
	triples = []
	for event in events:
		triple = get_triple_from_event(event)
		if len(triple) > 0:
			triples.append(triple)
	return triples
