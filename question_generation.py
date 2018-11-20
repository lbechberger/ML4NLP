import knowledgestore.ks as ks
import xml.etree.cElementTree as ET

def generate_natural_questions(triple):
    '''Generates a set of possible human-like questions for elements in a given RDF-style triple.'''
    for elem in triple:
        if 'PERSON' in elem:
            pass
        elif 'ORGANIZATION' in elem:
            pass
        elif 'LOCATION' in elem:
            pass
        elif 'MISC' in elem:
            pass

def generate_artificial_questions(triple):
    '''Generates a set of possible query-like questions for elements in a given RDF-style triple.'''
    for elem in triple:
        if 'PERSON' in elem:
            pass
        elif 'ORGANIZATION' in elem:
            pass
        elif 'LOCATION' in elem:
            pass
        elif 'MISC' in elem:
            pass

    
def write_into_xml(q_a_pairs, filename):
    '''Used to write the information that we generated into an XML file.'''
    pass


if __name__ == '__main__':
    n_q_a_pairs = [] # natural question-answer pairs
    a_q_a_pairs = [] # artificial question-answer pairs
    with open('triples.txt', 'r') as f:
        for triple in f:
            n_q_a_pairs.append(generate_natural_questions(triple))
            a_q_a_pairs.append(generate_artificial_questions(triple))
    f.close()
