# -*- coding: utf-8 -*-
"""
Interface to the knowledge store via REST API.

Created on Fri Oct 12 11:48:30 2018

@author: lbechberger
"""

import requests

"""
Runs the given SPARQL query and returns the results as a list of maps.

Each element of the resulting list contains a mapping from query variables to their binding.
If the SPARQL query does not return any results (because the result set is empty or because the query is broken),
this function returns an empty list.

For the query "SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 1", the result looks like this:
[{'s': 'http://en.wikinews.org/wiki/Bone_marrow_transplant_potentially_linked_to_cure_of_patient_with_AIDS#ev30'}]
"""
def run_sparql_query(query_string):
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/sparql', params={"query":query_string})
    json = req.json()
    if 'results' not in json.keys() or 'bindings' not in json['results'].keys():
        return []
    bindings = json['results']['bindings']   

    result = []
    for binding in bindings:
        local_map = {}
        for key, value in binding.items():
            local_map[key] = value['value']
        result.append(local_map)
    
    return result

"""
Queries the KnowledgeStore for the given property of the given mention and returns a list of results.

If either property or mention do not exist, an empty list is returned.

For the function call 'run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred")',
the result looks like this:
['require']
"""
def run_mention_query(mention_uri, prop):
    p = {"id":"<{0}>".format(mention_uri), "property":prop}
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/mentions', params=p)
    json = req.json()
    result = []
    
    if '@graph' in json.keys() and len(json['@graph']) >= 1 and prop in json['@graph'][0].keys():
        if isinstance(json['@graph'][0][prop], list):
            for element in json['@graph'][0][prop]:
                result.append(element['@id'])    
        elif '@value' in json['@graph'][0][prop].keys():
            result.append(json['@graph'][0][prop]['@value'])
    return result

"""
Queries the KnowledgeStore for the given property of the given mention and returns a list of results.

The function returns an empty list in case of invalid resource URI or invalid property.

For the function call 'run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:title")',
the result looks like this:
['Mexican president defends emigration']
"""
def run_resource_query(resource_uri, prop):
    p = {"id":"<{0}>".format(resource_uri), "property":prop}
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/resources', params=p)
    json = req.json()
    result = []
    
    if '@graph' in json.keys() and len(json['@graph']) >= 1 and prop in json['@graph'][0].keys():
        if isinstance(json['@graph'][0][prop], list):
            for element in json['@graph'][0][prop]:
                result.append(element['@id'])    
        elif '@value' in json['@graph'][0][prop].keys():
            result.append(json['@graph'][0][prop]['@value'])
    return result
    
"""
Retrieves the text of the news article stored under the given resource URI. Returns empty string for invalid resource URI.
"""
def run_files_query(resource_uri):
    p = {"id":"<{0}>".format(resource_uri)}
    req = requests.get('https://knowledgestore2.fbk.eu/nwr/wikinews/files', params=p)
    try:
        req.json()
        return ''
    except ValueError:    
        return req.text

"""
Converts a given mention URI into a resource URI by simply removing the suffix starting with "#".

For instance, the mention URI 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622' is 
transformed into the resource URI 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration'.

There is currently no sanity check on the input!
"""
def mention_uri_to_resource_uri(mention_uri):
    return mention_uri.split('#')[0]

"""
Short demo script showing how to use the API.
"""
def demo():
    print("1.) Run a SPARQL query:")
    print("-----------------------")
    print('run_sparql_query("SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 2")')
    print(run_sparql_query("SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 2"))
    
    print("\n2.) Run a query to get a certain property of a mention:")
    print("-------------------------------------------------------")
    print('run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred")')
    print(run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred"))
   
    print("\n3.) Run another SPARQL query to connect the mention to an event/entity:")
    print("-----------------------------------------------------------------------")
    print('run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622>}")')
    print(run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622>}"))
    
    print("\n4.) Run a query to get a certain property of a resource:")
    print("--------------------------------------------------------")
    print('run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:title")')
    print(run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:title"))

    print("\n5.) Run a query to get the original news article text:")
    print("------------------------------------------------------")
    print('run_files_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration")')
    print(run_files_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration"))

if __name__ == "__main__":
    demo()