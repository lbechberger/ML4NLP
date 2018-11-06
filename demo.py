# -*- coding: utf-8 -*-
"""
Examples for accessing the KnowledgeStore as shown in class

Created on Tue Nov  6 13:27:24 2018

@author: lbechberger
"""

import knowledgestore.ks as ks

query_string = "SELECT DISTINCT ?e WHERE {?m dbo:starring ?e . ?m dbo:genre dbpedia:Comedy . ?m dbo:starring dbpedia:Charlie_Sheen . ?e rdf:type dbo:PlayboyPlaymate . ?e dbo:birthPlace dbpedia:Canada}"
sparql_result = ks.run_sparql_query(query_string)
print(sparql_result)
print("")

graph_string = """SELECT ?label 
WHERE {
GRAPH <http://www.newsreader-project.eu/modules/dbpedia-en> 
{ dbpedia:Angela_Merkel rdfs:label ?label } 
}"""
graph_result = ks.run_sparql_query(graph_string)
print(graph_result)
print("")

merkel_string = """SELECT ?label 
WHERE {
dbpedia:Angela_Merkel rdfs:label ?label
}"""
print(ks.run_sparql_query(merkel_string))
print("")

get_graph_query = """SELECT ?label ?graph
WHERE {
GRAPH ?graph {dbpedia:Angela_Merkel rdfs:label ?label} 
}"""
print(ks.run_sparql_query(get_graph_query))
print("")

birth_location_query = """SELECT ?p 
WHERE {
{ dbpedia:Angela_Merkel dbo:birthPlace ?p } UNION { dbpedia:Willy_Brandt dbo:birthPlace ?p} }"""
print(ks.run_sparql_query(birth_location_query))
print("")


mention_result = ks.run_mention_query("http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover-up_was_reported_by_fake_SEAL#char=62,67", "nwr:pos")
print(mention_result)
print("")

resource_result = ks.run_resource_query("http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover-up_was_reported_by_fake_SEAL", "ks:hasMention")
print("")
print(len(resource_result))

text_result = ks.run_files_query("http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover-up_was_reported_by_fake_SEAL")
print(text_result)
print(len(text_result))