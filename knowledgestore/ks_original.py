# -*- coding: utf-8 -*-
"""
Interface to the knowledge store via REST API.

Created on Fri Oct 12 11:48:30 2018

@author: lbechberger
"""

import requests, os, pickle

def run_sparql_query(query_string):
    """
    Runs the given SPARQL query and returns the results as a list of maps.
    
    Each element of the resulting list contains a mapping from query variables to their binding.
    If the SPARQL query does not return any results (because the result set is empty or because the query is broken),
    this function returns an empty list.
    
    For the query "SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 1", the result looks like this:
    [{'s': 'http://en.wikinews.org/wiki/Bone_marrow_transplant_potentially_linked_to_cure_of_patient_with_AIDS#ev30'}]
    """
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/sparql', params={"query":query_string})
    
    try:    
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
    except ValueError:
        print("Warning! Couldn't parse JSON!")
        return []

def run_mention_query(mention_uri, prop):
    """
    Queries the KnowledgeStore for the given property of the given mention and returns a list of results.
    
    If either property or mention do not exist, an empty list is returned.
    
    For the function call 'run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred")',
    the result looks like this:
    ['require']
    """
    p = {"id":"<{0}>".format(mention_uri)}    
    if prop != "@type":
        p['property'] = prop
    
    req = requests.get('http://knowledgestore2.fbk.eu/nwr/wikinews/mentions', params=p)
    json = req.json()
    result = []
    
    if '@graph' in json.keys() and len(json['@graph']) >= 1 and prop in json['@graph'][0].keys():
        if isinstance(json['@graph'][0][prop], list):
            for element in json['@graph'][0][prop]:
                if '@id' in element:
                    result.append(element['@id'])
                else:
                    result.append(element)
        elif '@value' in json['@graph'][0][prop].keys():
            result.append(json['@graph'][0][prop]['@value'])
        elif '@id' in json['@graph'][0][prop].keys():
            result.append(json['@graph'][0][prop]['@id'])
    return result

def run_resource_query(resource_uri, prop):
    """
    Queries the KnowledgeStore for the given property of the given mention and returns a list of results.
    
    The function returns an empty list in case of invalid resource URI or invalid property.
    
    For the function call 'run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:title")',
    the result looks like this:
    ['Mexican president defends emigration']
    """
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
        elif '@id' in json['@graph'][0][prop].keys():
            result.append(json['@graph'][0][prop]['@id'])
    return result
    
def run_files_query(resource_uri):
    """
    Retrieves the text of the news article stored under the given resource URI. Returns empty string for invalid resource URI.
    """
    p = {"id":"<{0}>".format(resource_uri)}
    req = requests.get('https://knowledgestore2.fbk.eu/nwr/wikinews/files', params=p)
    try:
        req.json()
        return ''
    except ValueError:    
        return req.text

def mention_uri_to_resource_uri(mention_uri):
    """
    Converts a given mention URI into a resource URI by simply removing the suffix starting with "#".
    
    For instance, the mention URI 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622' is 
    transformed into the resource URI 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration'.
    
    There is currently no sanity check on the input!
    """
    return mention_uri.split('#')[0]


def get_all_resource_uris():
    """
    Returns a list which contains all resource URIs that are found in the data base.
    
    If possible, use locally cached version (in pickle file), if not download everything again (slow).
    """
    if os.path.isfile('resourceURIs.pickle'):
        with open('resourceURIs.pickle', "rb") as f:
            return pickle.load(f)
    else:
        sparql_query = "SELECT DISTINCT ?s WHERE { ?e gaf:denotedBy ?m . BIND(STRBEFORE(STR(?m), '#') AS ?s) }"
        sparql_result = run_sparql_query(sparql_query)
        result = []
        for entry in sparql_result:
            resource_uri = entry['s']
            if resource_uri not in result:
                result.append(resource_uri)
        with open('resourceURIs.pickle', 'wb') as f:
            pickle.dump(result, f)
        return result

def get_applicable_news_categories(resource_uri, category_names):
    """
    Checks which of the given category_names are applicable to the given news article (given by resource_uri).
    
    Crawls the wikinews website and searches for the category_uri in the original HTML code (which contains links to the categories). 
    Returns a list of applicable category names (subset of category_names).
    """

    req = requests.get(resource_uri)
    pattern = '<a href="/wiki/Category:{0}" title="Category:{1}">{1}</a>'
    result = []

    for category_name in category_names:
        if pattern.format(category_name.replace(' ', '_'), category_name.replace('_', ' ')) in req.text:
            result.append(category_name)   
    return result

top_level_category_names = ["Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business", 
                                "Education", "Environment", "Health", "Local only", "Media", "Obituaries", 
                                "Politics and conflicts", "Science and technology", "Sports", "Wackynews", "Weather", "Women"]   

def get_all_resource_category_mappings(category_names):
    """
    Returns a mapping of resource URIs to news categories. 
    
    If a precomputed mapping is found, this is used. Otherwise, it is dynamically recomputed and stored.
    """
    if os.path.isfile('resource_category_mappings.pickle'):
        with open('resource_category_mappings.pickle', "rb") as f:
            data = pickle.load(f)
            if data['category_names'] == category_names:
                return data['mappings']
    mappings = {}
    all_resource_uris = get_all_resource_uris()
    for resource_uri in all_resource_uris:
        mappings[resource_uri] = get_applicable_news_categories(resource_uri, category_names)
    result = {'category_names' : category_names, 'mappings' : mappings}
    with open('resource_category_mappings.pickle', 'wb') as f:
            pickle.dump(result, f)
    return mappings

def demo():
    """
    Short demo script showing how to use the API.
    """
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

    print("\n6.) Get a list of all news articles:")
    print("------------------------------------")
    print('len(get_all_resource_uris())')
    print(len(get_all_resource_uris()))
    
    print("\n7.) Check for applicable top level categories to a given article:")
    print("-----------------------------------------------------------------")
    print('get_applicable_news_categories("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", top_level_category_names)')
    print(get_applicable_news_categories("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", top_level_category_names))

    


if __name__ == "__main__":
    demo()