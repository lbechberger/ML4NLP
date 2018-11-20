# -*- coding: utf-8 -*-
"""
Solution to the Vips questions of session 04.

Created on Tue Nov 13 09:25:43 2018

@author: lbechberger
"""
import sys
sys.path.append(".")
import knowledgestore.ks as ks
import nltk

# first question
article_uri = "http://en.wikinews.org/wiki/'Worst_song_of_all_time'_becomes_YouTube_sensation"
mention_uris = ks.run_resource_query(article_uri, "ks:hasMention")

counter = 0
for mention_uri in mention_uris:
    types = ks.run_mention_query(mention_uri, "@type")
    if "nwr:RelationMention" in types:
        counter += 1
        
print("Answer for question 1: {0}".format(counter))


# second question
sparql_query = "SELECT DISTINCT ?m WHERE { dbpedia:Angela_Merkel gaf:denotedBy ?m }"
sparql_result = ks.run_sparql_query(sparql_query)

resource_uris = []
for binding in sparql_result:
    mention_uri = binding['m']
    resource_uri = ks.mention_uri_to_resource_uri(mention_uri)
    if resource_uri not in resource_uris:
        resource_uris.append(resource_uri)

counter = 0
all_mappings = ks.get_all_resource_category_mappings(ks.top_level_category_names)
for resource_uri in resource_uris:
    if "Economy and business" in all_mappings[resource_uri]:
        counter += 1

print("Answer for question 2: {0}".format(counter))

# third question
resource_uri = "http://en.wikinews.org/wiki/Christian_Wulff_elected_Germany's_new_president"
text = ks.run_files_query(resource_uri)
sentences = nltk.sent_tokenize(text)
sentence = sentences[0]
tokenized = nltk.word_tokenize(sentence)
pos_tagged = nltk.pos_tag(tokenized)
chunked = nltk.ne_chunk(pos_tagged)
print(chunked)