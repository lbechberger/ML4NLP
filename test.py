from knowledgestore import ks

query = """SELECT ?mention ?event (strbefore(str(?atime), "#") AS ?article)
WHERE {
?event rdf:type sem:Event .
?event sem:hasAtTime ?atime .
?event gaf:denotedBy ?mention
}
ORDER BY DESC(?article)
LIMIT 5"""

test_response = ks.run_sparql_query(query)

# test_mention = test[0]["mention"]

print(test_response)
