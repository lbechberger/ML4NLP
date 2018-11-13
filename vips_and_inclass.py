import ks

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

tmp = ks.get_all_resource_category_mappings(ks.top_level_category_names)
print(tmp)