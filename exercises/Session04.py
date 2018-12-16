# -*- coding: utf-8 -*-
"""
Examples for accessing the KnowledgeStore as shown in Session 04.

Created on Mon Nov 12 13:38:20 2018

@author: lbechberger
"""

import sys
sys.path.append(".")
import knowledgestore.ks as ks
import nltk

# get @type of given mention
mention_types = ks.run_mention_query("http://en.wikinews.org/wiki/German_president_dissolves_parliament;_elections_in_September#char=2081,2101", "@type")
print(mention_types)

# check applicable categories directly on given article
parliament_uri = "http://en.wikinews.org/wiki/German_president_dissolves_parliament;_elections_in_September"
klopp_uri = "http://en.wikinews.org/wiki/J%c3%bcrgen_Klopp_signs_3_year_contract_with_Liverpool"

my_categories = ["Politics and conflicts", "Sports", "Germany"]
print(ks.get_applicable_news_categories(parliament_uri, my_categories))
print(ks.get_applicable_news_categories(klopp_uri, my_categories))

# count all articles
num_articles = len(ks.get_all_resource_uris())
print(num_articles)

# use pre-stored knowledge about categories to count number of sports articles
all_mappings = ks.get_all_resource_category_mappings(ks.top_level_category_names)
num_sports_articles = 0

for resource_uri in ks.get_all_resource_uris():
    if "Sports" in all_mappings[resource_uri]:
        num_sports_articles += 1

print(num_sports_articles, num_sports_articles/num_articles)

# use NLTK to do named entity recognition
text = "John Wilkes Booth shot Abraham Lincoln. This did not happen inside the White House."
sentences = nltk.sent_tokenize(text)
print(sentences)

for sentence in sentences:
    print(sentence)
    word_tokenized = nltk.word_tokenize(sentence)
    print(word_tokenized)
    pos_tagged = nltk.pos_tag(word_tokenized)
    print(pos_tagged)
    ne_chunked = nltk.ne_chunk(pos_tagged)
    print(ne_chunked)