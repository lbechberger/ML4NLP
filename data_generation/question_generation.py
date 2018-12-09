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


def naively_generate_negatives(event_pos):
    """Generates negative examples by taking correct triples, checking other entities from a list of all entities and substituting one of the entities in the triple with another non-matching one from the list."""
    actual_agent = agents[event_pos]
    actual_patient = patients[event_pos]
    mode = random.randint(1,2)
    # @TODO: evaluate how random this actually is or if it generates a bias towards either subjects or objects
    same_as_before = True
    if mode == 1:
        while same_as_before:
            print(actual_agent)
            new_agent_id = random.randint(0, len(agents) - 1)
            replacement_agent = agents[new_agent_id]
            print(replacement_agent)
            if replacement_agent != actual_agent: 
                same_as_before = False
        # returns a dict that can be written into the csv file
        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": replacement_agent, "predicate": predicates[event_pos], "patient": patients[event_pos]}
    else:
        while same_as_before:
            new_patient_id = random.randint(0, len(patients) - 1)
            print(actual_patient)
            replacement_patient = patients[new_patient_id]
            print(replacement_patient)
            if replacement_patient != actual_patient:
                same_as_before = False

        return {"event_id": event_ids[event_pos], "article_id": article_ids[event_pos], "uri": uris[event_pos], "event": events[event_pos], "agent": agents[event_pos], "predicate": predicates[event_pos], "patient": replacement_patient}      


def smartly_generate_negatives(event_pos):
    """Generates negative examples by taking correct triples, checking other entities from the source text of a triple
    and substituting one of the entities in the triple with another non-matching one from the text."""
    pass


if __name__ == '__main__':
    # sets integer determining amount of negatives generated to a value determined by the user
    # if user omits a value, it is automatically set to 10
    if len(sys.argv) != 1:
        generation_factor = sys.argv[1]
    else:
        generation_factor = 10
    
    # read in the information from the csv file
    with open("demo_triples.csv") as csvDataFile:
        data_file = csv.reader(csvDataFile)
        for row in data_file:
            event_ids.append(row[0])
            article_ids.append(row[0])
            uris.append(row[0])
            events.append(row[0])
            agents.append(row[0])
            predicates.append(row[0])
            patients.append(row[0])
    
    # start writing a new csv file
    with open("negatives.csv", "w") as csvDataFile:
        writer = csv.DictWriter(csvDataFile, fieldnames=fieldnames)
        writer.writeheader()
        for event_pos in range(len(events)):
            for n in range(int(generation_factor)):
                writer.writerow(naively_generate_negatives(event_pos))