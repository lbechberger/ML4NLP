import nltk
import knowledgestore.ks as ks
import csv

fieldnames = ["event_id", "article_id", "uri", "event", "agent", "agent_NER", "agent_POS", "agent_RELPOS", "predicate", "predicate_POS", "predicate_RELPOS", "patient", "patient_NER", "patient_POS", "patient_RELPOS"]
event_ids = []
article_ids = []
articles = []
uris = []
events = []
agents = []
predicates = []
patients = []
article_ranges = []
agent_ner = []
agent_pos = []
agent_relpos = []
predicate_pos = []
predicate_relpos = []
patient_ner = []
patient_pos = []
patient_relpos = []


def find_source_sent(triple_index, source, window):
    """
    Finds the source sentence that included the event mention.
    """
    # two-step approach
    # step 1: find the mention in the target text
    query_string = "SELECT ?s WHERE {<http://en.wikinews.org/wiki/" + events[triple_index] + "> gaf:denotedBy ?s}"
    print(query_string)
    denotation = ks.run_sparql_query(query_string)
    # remove all entries from other articles (sometimes necessarry because the data in the Knowledgestore sucks severely)
    for index in range(len(denotation)):
        if uris[triple_index] not in denotation[index]['s']:
            print("Removed " + denotation[index]['s'])
            denotation[index] = "delete this"
    denotation.remove("delete this")
    for index in range(len(denotation)):
        print(denotation[index]['s'])

    # set 28 as beginning to find because the first 28 characters in any URI are the address to wikinews
    # absolute_wordpos = denotation.find("#char=", 28)
    # print(absolute_wordpos)
    # step 2: isolate the full sentence

    # filler return statement
    return "You are an idiot."


def extract_features(triple_index, sent):
    """
    Extracts the following features for an entry in our data set: NER, POS, relative position.
    """
    agent = agents[triple_index].replace('+', ' ')
    pos_tagged_sent = nltk.pos_tag(nltk.word_tokenize(sent))
    ner_tagged_sent = nltk.ne_chunk(pos_tagged_sent)
    # @TODO: implement NER
    # @TODO: implement POS
    # @TODO: implement relative position
    pass


if __name__ == '__main__':
    # read in the information from the csv file
    with open("demo_triples.csv") as csv_data_file:
        data_file = csv.reader(csv_data_file, delimiter=';')
        index = 0
        temp_lower_bound = 0
        temp_upper_bound = 0
        for row in data_file:
            event_ids.append(row[0])
            article_ids.append(row[1])
            # keep track of article ranges
            if index != 0:
                if row[1] != article_ids[index - 1]:
                    temp_upper_bound = index - 1
                    article_ranges.append([temp_lower_bound, temp_upper_bound])
                    temp_lower_bound = index
            uris.append(row[2])
            events.append(row[3])
            agents.append(row[4])
            predicates.append(row[5])
            patients.append(row[6])
            index += 1
        temp_upper_bound = index - 1
        article_ranges.append([temp_lower_bound, temp_upper_bound])

    index = 0
    # for ar in article_ranges:
    #     article_uri = uris[ar[0]]
    #     print("Retrieving article text for "+ article_uri)
    #     article = ks.run_files_query(article_uri)
    #     while index in range(ar[1]):
    #         sent = find_source_sent(index, article, 10)
    #         extract_features(index, sent)
    #         index += 1
    article = ks.run_files_query(uris[0])
    sent = find_source_sent(0, article, 10)

    # start writing a new csv file
    with open("data_with_features.csv", "w") as csv_data_file:
        writer = csv.DictWriter(csv_data_file, fieldnames=fieldnames)
        writer.writeheader()
        pass