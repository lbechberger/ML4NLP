import nltk
import knowledgestore.ks as ks
import knowledgestore.ks_extensions as ks2
from nltk.sem.relextract import extract_rels, rtuple
import re
import numpy as np
from article_store import ArticleCache
import shelve


def main():
    article_cache = ArticleCache()
    entity_set = EntitySet()

    for article, uri in article_cache.generate_articles_and_uris(verbose=True):
        mentions = ks.run_resource_query(uri, 'ks:hasMention')
        tmp = list(filter(lambda x: x is not None, [get_entity(article, mention, entity_set) for mention in mentions]))
        print(tmp)
    import sys;
    sys.exit(1)


def main2():
    article_cache = ArticleCache()
    entity_set = EntitySet()

    for article, uri in article_cache.generate_articles_and_uris(verbose=True):
        mentions = ks.run_resource_query(uri, 'ks:hasMention')
        tmp = list(filter(lambda x: x is not None, [get_entity(article, mention, entity_set) for mention in mentions]))
        print(tmp)
    import sys;
    sys.exit(1)

    # article_cache.cache_all_articles()
    # import sys; sys.exit(1)
    # for article in article_cache.generate_articles(verbose=True):
    #     pos_tags = get_sentences(article)
    #     print(per_of_gpe(pos_tags))
    uri = "http://en.wikinews.org/wiki/SEALs_say_US_officer's_cover-up_was_reported_by_fake_SEAL"
    mentions = ks.run_resource_query(uri, 'ks:hasMention')
    article = article_cache[uri]
    # print('EVENTS:')
    # for mention in mentions:
    #     entity = get_event(article, mention)
    #     if entity:
    #         print(entity)
    entities = [get_entity(article, mention, entity_set) for mention in mentions if
                get_entity(article, mention, entity_set)]
    print('ENTITIES:')
    for mention in mentions:
        entity = get_entity(article, mention, entity_set)
        if entity:
            print(entity)

        # positions = [int(i) for i in mention.split('#')[1].split('=')[1].split(',')]
        # print('Going for:', article[positions[0]:positions[1]]) #TODO look these up in a structured knowledgebase, creating a gazetteer?
        # print(ks.run_mention_query(mention, 'nwr:pred')) #interesting, for simple verbs&nouns this is the stem
        # print(get_e(mention))


def get_word_and_sentence(article, mention):
    positions = [int(i) for i in mention.split('#')[1].split('=')[1].split(',')]
    sentences_and_positions = list(
        zip(nltk.sent_tokenize(article), list(np.cumsum(np.array([len(i) for i in nltk.sent_tokenize(article)])))))
    try:
        correct_sent = list(filter(lambda x: x[1] > positions[0], sentences_and_positions))[0][0]
    except:
        return None, None
    return article[positions[0]:positions[1]], correct_sent


def get_event(article, mention):
    tmp = get_e(mention)
    if tmp and 'ev' in tmp.split('#')[-1]:
        word, sent = get_word_and_sentence(article, mention)
        if sent:
            return 'Event:', word, tmp, sent


def get_entity(article, mention, entity_set):
    tmp = get_e(mention)
    if tmp and 'http://dbpedia.org/resource/' in tmp:
        word, sent = get_word_and_sentence(article, mention)
        stem = ks.run_mention_query(mention, 'nwr:pred')
        additional_info = None #FIXPRECOMMIT # additional_info = entity_set[tmp]
        if sent:
            return {'stem': stem, 'word': word, 'dbpedia': tmp.replace('http://dbpedia.org/resource/', ''),
                    'sent': sent, 'info': additional_info}


class EntitySet():
    def __init__(self, cache_file='entity_cache.shelve'):
        self.cache_file = cache_file
        self.prefix_dict = {'dbp': 'http://dbpedia.org/ontology/', 'dc': 'http://purl.org/dc/elements/1.1/',
                            'gaf': 'http://groundedannotationframework.org/gaf#',
                            'w3': 'http://www.w3.org/2000/01/rdf-schema#'}

    def __getitem__(self, entity):
        with shelve.open(self.cache_file) as cache:
            if entity not in cache:
                cache[entity] = self.get_entity_info(entity)
            else:
                return cache[entity]

    def get_entity_info(self, entity):
        entity_info = {}
        entity = entity.replace('http://dbpedia.org/resource/', 'dbpedia:')
        entity_info['types'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' rdf:type ?l}') \
                               + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dc:description ?l }',
                                                        self.prefix_dict)
        if 'http://dbpedia.org/ontology/Person' in entity_info['types']:
            entity_info = {**entity_info, **self.get_person_entity_info(entity)}
        elif 'http://dbpedia.org/ontology/Event' in entity_info['types']:
            entity_info = {**entity_info, **self.get_event_entity_info(entity)}
        entity_info['mentioned_in'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' gaf:denotedBy ?l }',
                                                             self.prefix_dict)
        entity_info['comment'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' w3:comment ?l }',
                                                        self.prefix_dict)
        return entity_info

    def get_person_entity_info(self, entity):
        entity_info = {}
        entity_info['altnames'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' rdfs:label ?l}') \
                                  + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:alias ?l }',
                                                           self.prefix_dict) \
                                  + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:alternativeName ?l }',
                                                           self.prefix_dict) \
                                  + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:birthName ?l }',
                                                           self.prefix_dict)
        entity_info['birtdate'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:birthDate ?l }',
                                                         self.prefix_dict) \
                                  + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:birthYear ?l }',
                                                           self.prefix_dict)
        entity_info['birtplace'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:birthPlace ?l }',
                                                          self.prefix_dict)
        entity_info['country'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:country ?l }',
                                                        self.prefix_dict) \
                                 + ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:nationality ?l }',
                                                          self.prefix_dict)
        entity_info['gender'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:gender ?l }',
                                                       self.prefix_dict)
        entity_info['profession'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:profession ?l }',
                                                           self.prefix_dict)
        return entity_info

    def get_event_entity_info(self, entity):
        entity_info = {}
        entity_info['date'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:date ?l }', self.prefix_dict)
        entity_info['partof'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:isPartOf ?l }',
                                                       self.prefix_dict)
        entity_info['place'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:place ?l }',
                                                      self.prefix_dict)
        entity_info['startdate'] = ks2.run_dictless_query('SELECT ?l WHERE {' + entity + ' dbp:startDate ?l }',
                                                          self.prefix_dict)
        return entity_info


def get_e(mention):
    '''get entities/events given a mention'''
    res = ks.run_sparql_query('SELECT ?e WHERE {?e gaf:denotedBy <' + mention + '>}')
    return None if not res else res[0]['e']


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


if __name__ == '__main__':
    main()