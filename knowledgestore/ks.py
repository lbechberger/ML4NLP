# -*- coding: utf-8 -*-
"""
Interface to the knowledge store via REST API.

Created on Fri Oct 12 11:48:30 2018

@author: lbechberger
"""

import requests


def run_sparql_query(query_string):
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/sparql', params={"query":query_string})
    return req.json()

def run_mention_query(mention_uri, prop):
    p = {"id":"<{0}>".format(mention_uri), "property":prop}
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/mentions', params=p)
    return req.json()

def run_resource_query(resource_uri):
    p = {"id":"<{0}>".format(resource_uri)}
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/resources', params=p)
    return req.json()

def run_files_query(resource_uri):
    p = {"id":"<{0}>".format(resource_uri)}
    req = requests.get('https://knowledgestore2.fbk.eu/nwr/wikinews/files', params=p)
    return req.text

def demo():
    print("Run a SPARQL query and get the result as JSON dictionary:")
    print(run_sparql_query("SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 10"))
    
    print("\nRun a query to get a certain property of a mention (result is a JSON dictionary):")
    print(run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred"))
   
    print("\nRun another SPARQL query to connect the mention to an event/entity:")
    print(run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622>}"))
    
    print("\nRun a query to get a certain property of a resource (result is a JSON dictionary):")
    print(run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration"))

    print("\nRun a query to get the original news article text as a string:")
    print(run_files_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration"))

if __name__ == "__main__":
    demo()