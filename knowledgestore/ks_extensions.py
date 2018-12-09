import knowledgestore.ks as ks

def run_dictless_query(query, prefix_dict = None):
    variable_names = query.split("SELECT")[1].split("WHERE")[0].strip()
    assert len(variable_names.split(","))
    if prefix_dict:
        prefix_str = "".join(["PREFIX "+key+": <"+value+"> \n" for key, value in list(prefix_dict.items())])
        query = prefix_str + query
    res = ks.run_sparql_query(query)
    return [i[variable_names[1:]] for i in res]

