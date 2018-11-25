import knowledgestore.ks as ks
import nltk, re
from nltk.sem.relextract import extract_rels, rtuple
import pandas as pd 

## NER
# 1. 'ORG'
# 2. 'LOC'
# 3. 'GPE'

## define relation
OF = re.compile(r'.*\bof\b.*')
IN = re.compile(r'.*\bin\b(?!\b.+ing)')

def main():
    run(100)


## given article, tokenlize to sentences, words and POS tags
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    # print(len(sentences), "sentences")
    return sentences	

def per_of_gpe(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=OF)
        for rel in rels:
            if rel != []:
                print("per_of_gpe")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def org_of_gpe(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('ORG', 'GPE', sent, corpus='ace', pattern=OF)
        for rel in rels:
            if rel != []:
                print("org_of_gpe")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def per_of_loc(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'LOC', sent, corpus='ace', pattern=OF)
        for rel in rels:
            if rel != []:
                print("per_in_loc")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def org_of_loc(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('ORG', 'LOC', sent, corpus='ace', pattern=OF)
        for rel in rels:
            if rel != []:
                print("org_in_loc")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def per_in_gpe(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=IN)
        for rel in rels:
            if rel != []:
                print("per_in_gpe")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def org_in_gpe(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('ORG', 'GPE', sent, corpus='ace', pattern=IN)
        for rel in rels:
            if rel != []:
                print("org_in_gpe")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def per_in_loc(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'LOC', sent, corpus='ace', pattern=IN)
        for rel in rels:
            if rel != []:
                print("per_in_loc")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)

def org_in_loc(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('ORG', 'LOC', sent, corpus='ace', pattern=IN)
        for rel in rels:
            if rel != []:
                print("org_in_loc")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)


def find_relation(pos_tags):
    per_of_gpe(pos_tags)
    # org_of_gpe(pos_tags)
    per_of_loc(pos_tags)
    # org_of_loc(pos_tags)
    per_in_gpe(pos_tags)
    # org_in_gpe(pos_tags)
    per_in_loc(pos_tags)
    # org_in_loc(pos_tags)
    # org_in_gpe(pos_tags)


def run(num_article):
    # get uri
    all_uris = pd.read_csv("all_article_uris.csv")
    # get POS tags given an article
    for uri in all_uris.article[:num_article]:
        result = ks.run_files_query(uri)
        pos_tags = get_sentences(result)
        find_relation(pos_tags)


if __name__ == '__main__':
    main()