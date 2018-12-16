import knowledgestore.ks as ks
import csv
import sys
import random


fieldnames = ["event_id", "article_id", "uri", "event", "agent", "predicate", "patient"]
event_ids = []
article_ids = []
uris = []
events = []
agents = []
predicates = []
patients = []
article_ranges = []


def naively_generate_negatives(event_pos):
    """Generates negative examples by taking correct triples, checking other entities from a list of all entities and substituting one of the entities in the triple with another non-matching one from the list."""
    actual_agent = agents[event_pos]
    actual_patient = patients[event_pos]
    mode = random.randint(1,2)
    # @TODO: evaluate how random this actually is or if it generates a bias towards either subjects or objects
    same_as_before = True
    # have random decision whether to replace the agent or the patient
    if mode == 1:
        # randomly pick out an agent out of a list of all agents until it is not the same as the original agent
        while same_as_before:
            new_agent_id = random.randint(0, len(agents) - 1)
            replacement_agent = agents[new_agent_id]
            if replacement_agent != actual_agent: 
                same_as_before = False
        # returns a dict that can be written into the csv file
        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": replacement_agent, "predicate": predicates[event_pos], "patient": patients[event_pos]}
    else:
        # randomly pick out a patient out of a list of all patients until it is not the same as the original patient
        while same_as_before:
            new_patient_id = random.randint(0, len(patients) - 1)
            replacement_patient = patients[new_patient_id]
            if replacement_patient != actual_patient:
                same_as_before = False

        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": agents[event_pos], "predicate": predicates[event_pos], "patient": replacement_patient}      


def smartly_generate_negatives(event_pos):
    """Generates negative examples by taking correct triples, checking other entities from the source text of a triple
    and substituting one of the entities in the triple with another non-matching one from the text."""
    actual_agent = agents[event_pos]
    actual_patient = patients[event_pos]
    article_id_pos = article_ids[event_pos] - 1
    lower_bound = article_ranges[article_id_pos][0]
    upper_bound = article_ranges[article_id_pos][1]
    mode = random.randint(1,2)
    # @TODO: evaluate how random this actually is or if it generates a bias towards either subjects or objects
    same_as_before = True
    # have random decision whether to replace the agent or the patient
    if mode == 1:
        # randomly pick out an agent out of a list of all agents until it is not the same as the original agent
        while same_as_before:
            new_agent_id = random.randint(lower_bound, upper_bound)
            replacement_agent = agents[new_agent_id]
            # added condition to check for similarity and try if it is still similar when patient from that triple is used instead
            if replacement_agent == actual_agent:
                replacement_agent = patients[new_agent_id]
            if replacement_agent != actual_agent: 
                same_as_before = False
        # returns a dict that can be written into the csv file
        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": replacement_agent, "predicate": predicates[event_pos], "patient": patients[event_pos]}
    else:
        # randomly pick out a patient out of a list of all patients until it is not the same as the original patient
        while same_as_before:
            new_patient_id = random.randint(lower_bound, upper_bound)
            replacement_patient = patients[new_patient_id]
            # see comment above
            if replacement_patient == actual_patient:
                replacement_patient = agents[new_patient_id]
            if replacement_patient != actual_patient:
                same_as_before = False

        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": agents[event_pos], "predicate": predicates[event_pos], "patient": replacement_patient}      



if __name__ == '__main__':
    # sets integer determining amount of negatives generated to a value determined by the user
    # if user omits a value, it is automatically set to 10
    if len(sys.argv) != 1:
        generation_factor = sys.argv[1]
    else:
        generation_factor = 10
    
    # read in the information from the csv file
    with open("demo_triples.csv") as csvDataFile:
        data_file = csv.reader(csvDataFile, delimiter=';')
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
    
    # start writing a new csv file
    with open("negatives.csv", "w") as csvDataFile:
        writer = csv.DictWriter(csvDataFile, fieldnames=fieldnames)
        writer.writeheader()
        # for each positive, generate n negatives, where n is determined by the user (10 by default, should the user not care)
        for event_pos in range(len(events)):
            for n in range(int(generation_factor)):
                writer.writerow(naively_generate_negatives(event_pos))