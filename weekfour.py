import nltk
import knowledgestore.ks as ks
from nltk.sem.relextract import extract_rels, rtuple
import re
import numpy as np
from article_store import ArticleCache

## define relation
OF = re.compile(r'.*\bof\b.*')
IN = re.compile(r'.*\bin\b(?!\b.+ing)')


def main():
    article_cache = ArticleCache()
    # article_cache.cache_all_articles()
    # for article in article_cache.generate_articles(verbose=True):
    #     pos_tags = get_sentences(article)
    #     print(per_of_gpe(pos_tags))
    uri = "http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover-up_was_reported_by_fake_SEAL"
    mentions = ks.run_resource_query(uri, 'ks:hasMention')
    article = article_cache[uri]
    print("EVENTS:")
    for mention in mentions:
        entity = get_event(article, mention)
        if entity:
            print(entity)
    print("ENTITIES:")
    for mention in mentions:
        entity = get_entity(article, mention)
        if entity:
            print(entity)

        # positions = [int(i) for i in mention.split("#")[1].split("=")[1].split(",")]
        # print("Going for:", article[positions[0]:positions[1]]) #TODO look these up in a structured knowledgebase, creating a gazetteer?
        # print(ks.run_mention_query(mention, 'nwr:pred')) #interesting, for simple verbs&nouns this is the stem
        # print(get_e(mention))

def get_word_and_sentence(article, mention):
    positions = [int(i) for i in mention.split("#")[1].split("=")[1].split(",")]
    sentences_and_positions = list(zip(nltk.sent_tokenize(article), list(np.cumsum(np.array([len(i) for i in nltk.sent_tokenize(article)])))))
    try:
        correct_sent = list(filter(lambda x: x[1] > positions[0], sentences_and_positions))[0][0]
    except:
        return None, None
    return article[positions[0]:positions[1]], correct_sent


def get_event(article, mention):
    tmp = get_e(mention)
    if tmp and "ev" in tmp.split("#")[-1]:
        word, sent = get_word_and_sentence(article, mention)
        if sent:
            return "Event:", word, tmp, sent


def get_entity(article, mention):
    tmp = get_e(mention)
    if tmp and "http://dbpedia.org/resource/" in tmp:
        word, sent = get_word_and_sentence(article, mention)
        stem = ks.run_mention_query(mention, 'nwr:pred')
        if sent:
            return "Entity:", stem or word, tmp.replace("http://dbpedia.org/resource/", ""), sent


def get_e(mention):
    """get entities/events given a mention"""
    res = ks.run_sparql_query("SELECT ?e WHERE {?e gaf:denotedBy <" + mention + ">}")
    return None if not res else res[0]['e']


def per_of_gpe(pos_tags):
    for i, sent in enumerate(pos_tags):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=OF)
        for rel in rels:
            if rel != []:
                print("per_of_gpe")
                print('{0:<5}{1}'.format(i, rtuple(rel)))
                return rtuple(rel)









def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences





if __name__ == '__main__':
    main()