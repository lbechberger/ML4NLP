import ks
import nltk

########################################################### vips-questions 12.11.2018 ############################################################
# 1
# tmp = ks.run_resource_query("http://en.wikinews.org/wiki/Merkel:_Georgia_will_join_NATO", "ks:hasMention")
# print(len(tmp))

# 2
# tmp = ks.run_mention_query("http://en.wikinews.org/wiki/World_leaders_react_to_Obama_win#char=1881,1887", "nwr:propbankRef")
# print(tmp)

# 3
# tmp = ks.run_files_query("http://en.wikinews.org/wiki/World_leaders_react_to_Obama_win")
# print(len(tmp))

############################################################ session 13.11.2018 ###################################################################

# tmp = ks.run_mention_query("http://en.wikinews.org/wiki/German_president_dissolves_parliament;_elections_in_September#char=2081,2101", "@type")
# print(tmp)

# cats = ["Politics and conflicts", "Germany", "Sports"]
# tmp = ks.get_applicable_news_categories("http://en.wikinews.org/wiki/German_president_dissolves_parliament;_elections_in_September", cats)
# print(tmp)
# tmp = ks.get_applicable_news_categories("http://en.wikinews.org/wiki/J%c3%bcrgen_Klopp_signs_3_year_contract_with_Liverpool", cats)
# print(tmp)
#
# print(len(ks.get_all_resource_uris()))

# tmp = ks.get_all_resource_category_mappings(ks.top_level_category_names)
# print(tmp)
# print(len(tmp))

########################################################### vips-questions 14.11.2018 ############################################################

# tmp = ks.run_resource_query("http://en.wikinews.org/wiki/'Worst_song_of_all_time'_becomes_YouTube_sensation", "ks:hasMention")
# sum = [1 for mention in tmp if "nwr:RelationMention" in ks.run_mention_query(mention, "@type")]
# print(len(sum))


merkel_mentioned = [list(val.values())[0] for val in ks.run_sparql_query("SELECT ?x WHERE {dbpedia:Angela_Merkel gaf:denotedBy ?x}")]
merkel_articles = {ks.mention_uri_to_resource_uri(i) for i in merkel_mentioned}
tmp = ks.get_all_resource_category_mappings(ks.top_level_category_names)
articles_in_ecbusiness = {key for key,val in tmp.items() if "Economy and business" in val}
print(len(articles_in_ecbusiness & merkel_articles))
print(len([1 for art in merkel_articles if ks.get_applicable_news_categories(art, ["Economy and business"])])) #sanity-check


# tmp = ks.run_files_query("http://en.wikinews.org/wiki/Christian_Wulff_elected_Germany's_new_president")
# sentences = nltk.sent_tokenize(tmp)
# print(sentences[0])
# word_tokenized = nltk.word_tokenize(sentences[0])
# pos_tagged = nltk.pos_tag(word_tokenized)
# ne_chunked = nltk.ne_chunk(pos_tagged)
# print(ne_chunked)