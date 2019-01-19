import nltk
import knowledgestore.ks as ks
import csv

fieldnames = ["event_id", "article_id", "uri", "event", "agent", "text", "agent_NER", "agent_POS", "agent_RELPOS", "predicate", "predicate_POS", "predicate_RELPOS", "patient", "patient_NER", "patient_POS", "patient_RELPOS"]
event_ids = []
article_ids = []
articles = []
uris = []
events = []
agents = []
predicates = []
patients = []
text = []
article_ranges = []
agent_ner = []
agent_pos = []
agent_relpos = []
predicate_pos = []
predicate_relpos = []
patient_ner = []
patient_pos = []
patient_relpos = []


def extract_features(triple_index):
    """
    Extracts the following features for an entry in our data set: NER, POS, relative position.
    """
    agent = agents[triple_index].replace('+', ' ').replace('_', ' ')
    patient = patients[triple_index].replace('+', ' ').replace('_', ' ')
    predicate = predicates[triple_index].replace('+', ' ').replace('_', ' ')
    sent = nltk.word_tokenize(text[triple_index])

    # determine position(s) of agent (given the fact that an agent may be multiple words long)
    if ' ' in agent:
        agent = nltk.word_tokenize(agent)
        patient = nltk.word_tokenize(patient)
        predicate = nltk.word_tokenize(predicate)

        # find sequence of agent tokens in source sentence
        # previous_index = -1
        agent_token_indices = []
        patient_token_indices = []
        predicate_token_indices = []
        lower_sent = [word.lower() for word in sent]
        for token in agent:
            try:
                agent_token_indices.append(lower_sent.index(token.lower()))
            except ValueError:
                pass
            # check on whether tokens actually follow one another
            # if token_index != -1:
            #     # if this is the first token or token order so far is correct, update information accordingly
            #     # if token order so far is correct, proceed normally
            #     if previous_index = -1 or previous_index = token_index -1:
            #         previous_index = token_index
            #     else:
        for token in patient:
            try:
                patient_token_indices.append(lower_sent.index(token.lower()))
            except ValueError:
                pass
        for token in predicate:
            for index in range(len(sent)):
                if token.lower() in sent[index].lower():
                    predicate_token_indices.append(index)
        # only append relpos list if position was in the sentence
        if agent_token_indices:
            agent_relpos.append(agent_token_indices[-1])
        else:
            agent_relpos.append(-1)
        if patient_relpos:
            patient_relpos.append(patient_token_indices[0])
        else:
            patient_relpos.append(-1)
        if len(predicate_token_indices) > 1:
            predicate_relpos_temp = [predicate_token_indices[0], predicate_token_indices[-1]]
            predicate_relpos.append(predicate_relpos_temp)
        elif len(predicate_token_indices) == 1:
            predicate_relpos.append(predicate_token_indices[0])
        else:
            predicate_relpos.append(-1)

    temp_agent_pos = []
    temp_patient_pos = []
    temp_predicate_pos = []
    pos_tagged_sent = nltk.pos_tag(sent)
    for index in agent_token_indices:
        temp_agent_pos.append(pos_tagged_sent[index][1])
    agent_pos.append(temp_agent_pos)
    for index in predicate_token_indices:
        temp_predicate_pos.append(pos_tagged_sent[index][1])
    predicate_pos.append(temp_predicate_pos)
    for index in patient_token_indices:
        temp_patient_pos.append(pos_tagged_sent[index][1])
    patient_pos.append(temp_patient_pos)

    ner_tagged_sent = nltk.ne_chunk(pos_tagged_sent)
    for chunk in ner_tagged_sent:
        # filter out non-named entities
        if hasattr(chunk, "label"):
            match_detected = False
            # match named entities with agents, patients or predicate
            for word in chunk:
                # print(word)
                for token in agent:
                    if token in word:
                        match_detected = True
                        agents[triple_index] += ("\/" + chunk.label())
                        break
                for token in patient:
                    if token in word:
                        match_detected = True
                        patients[triple_index] += ("\/" + chunk.label())
                        break
                if match_detected:
                    break


if __name__ == '__main__':
    # read in the information from the csv file
    with open("demo_triples.csv") as csv_data_file:
        data_file = csv.reader(csv_data_file, delimiter=';')
        for row in data_file:
            # prune of all rows that do not contain valid information
            valid = True
            for entry in row:
                if not entry:
                    valid = False
                    break
            if valid:
                event_ids.append(row[0])
                # print("Event ID:" + row[0])
                article_ids.append(row[1])
                # print("Article ID:" + row[1])
                uris.append(row[2])
                # print("URI: " + row[2])
                events.append(row[3])
                # print("Event: " + row[3])
                agents.append(row[4])
                # print("Agent: " + row[4])
                predicates.append(row[5])
                # print("Predicate: " + row[5])
                patients.append(row[6])
                # print("Patient: " + row[6])
                text.append(row[7])
                # print("Text: " + row[7])

    for index in range(len(event_ids)):
        extract_features(index)

    # start writing a new csv file
    with open("data_with_features.csv", "w") as csv_data_file:
        writer = csv.DictWriter(csv_data_file, fieldnames=fieldnames)
        writer.writeheader()
        for event_pos in range(len(event_ids)):
            if r"\/" in agents[event_pos]:
                agent_ner = agents[event_pos].split(r"\/")[1]
                agents[event_pos] = agents[event_pos].split(r"\/")[0]
            else: agent_ner = "None"
            if r"\/" in patients[event_pos]:
                patient_ner = patients[event_pos].split(r"\/")[1]
                agents[event_pos] = agents[event_pos].split(r"\/")[0]
            row = {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": agents[event_pos], "text": text[event_pos], "agent_NER": agent_ner, "agent_POS": agent_pos[event_pos], "agent_RELPOS": agent_relpos[event_pos], "predicate": predicates[event_pos], "predicate_POS": predicate_pos[event_pos], "predicate_RELPOS": predicate_relpos[event_pos], "patient": patients[event_pos], "patient_NER": patient_ner, "patient_POS": patient_pos[event_pos], "patient_RELPOS": patient_relpos[event_pos]}
            writer.writerow(row)