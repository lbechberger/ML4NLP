import knowledgestore.ks as ks

# query_string = "SELECT DISTINCT ..."
# sparql_result = ks.run_sparql_query(query_string)
# print(sparql_result)


print(ks.run_resource_query('http://en.wikinews.org/wiki/Merkel:_Georgia_will_join_NATO%22', "ks:hasMention"))

tmp = ks.run_resource_query("http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover­up_was_reported_by_fake_SEAL", "ks:hasMention")
print(tmp)


query = """SELECT ?p
WHERE {
{ dbpedia:Angela_Merkel dbo:birthPlace ?p } UNION
{ dbpedia:Willy_Brandt dbo:birthPlace ?p}
}"""


tmp = ks.run_resource_query(query)
print(tmp)
