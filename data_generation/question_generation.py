import knowledgestore.ks as ks
from lxml import etree as ET

def generate_natural_questions(triple):
    '''Generates a set of possible human-like questions for elements in a given RDF-style triple.'''
    #@TODO: implement this function and alter write_into_xml so that it is compatible
    pass

def generate_artificial_questions(triple):
    '''Generates a set of possible query-like questions for elements in a given RDF-style triple.'''
    questions = []
    triple_temp = triple.split()
    questions.append("(?, " + triple_temp[1] + ", " + triple_temp[2] + ")")
    questions.append("(" + triple_temp[0] + ", ?, " + triple_temp[2] + ")")
    questions.append("(" + triple_temp[0] + ", " + triple_temp[1] + ", ?)")
    return questions

    
def write_into_xml(q_a_pairs, triples, filename):
    '''Used to write the information that we generated into an XML file.'''
    dataset = ET.Element("dataset")
    levels = []
    for index in range(len(triples)):
        levels.append(ET.SubElement(dataset, "qa_pair_" + str(index)))
        ET.SubElement(levels[index], "triple").text = item
        ET.SubElement(levels[index], "subject_question").text = q_a_pairs[index][0]
        ET.SubElement(levels[index], "relation_question").text = q_a_pairs[index][1]
        ET.SubElement(levels[index], "object_question").text = q_a_pairs[index][2]
    tree = ET.ElementTree(dataset)
    tree.write(filename + ".xml", pretty_print=True)


if __name__ == '__main__':
    n_q_a_pairs = [] # natural question-answer pairs
    a_q_a_pairs = [] # artificial question-answer pairs
    # with open('triples.txt', 'r') as f:
    #     for triple in f:
    #         n_q_a_pairs.append(generate_natural_questions(triple))
    #         a_q_a_pairs.append(generate_artificial_questions(triple))
    # f.close()

    questions = ["Barack_Obama father_of Malia_Obama", "Michelle_Obama mother_of Malia_Obama"]
    for item in questions:
        print("Generating questions...")
        n_q_a_pairs.append(generate_artificial_questions(item))
    write_into_xml(n_q_a_pairs, questions, "test")