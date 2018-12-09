import knowledgestore.ks as ks
import nltk


def tokenize_text(articles):
    """Processes the raw strings of the wikinews articles for further use."""
    sentences = []
    n = 1
    for article in articles:
        sentences = [nltk.word_tokenize(sentence) for sentence in (nltk.sent_tokenize(article))]
        print("Article number " + str(n) + "tokenized into sentences successfully!")
        n += 1
    return sentences


def pos_tag(sentences):
    """Generates POS tags for the text."""
    pos_tags = [nltk.pos_tag(sentence) for sentence in sentences]
    return pos_tags


def recognize_ne(sentences):
    pers_list = []
    org_list = []
    loc_list = []
    misc_list = []
    n = 1
    for sentence in sentences:
        print("Evaluating sentence number " + str(n) + ".")
        for elem in nltk.ne_chunk(sentence):
            if 'person' in elem.lower():
                pers_list.append(elem)
            elif 'organization' in elem.lower():
                org_list.append(elem)
            elif 'locaction' in elem.lower():
                loc_list.append(elem)
            elif 'misc' in elem.lower():
                misc_list.append(elem)
        n += 1
    return [pers_list, org_list, loc_list, misc_list]


def print_to_txt(triples):
    with open('triples.txt', 'w') as f:
        for triple in triples:
            f.write(triple[0] + " " + triple[1] + " " + triple[2] + "\n")
        f.close()


if __name__ == '__main__':
    wikinews_uris = ks.get_all_resource_uris()
    print("URIs retrieved successfully!")
    articles = []
    n = 1
    for uri in wikinews_uris: 
        articles.append(ks.run_files_query(uri))
        print("Retrieved content for article " + str(n))
        
        n += 1
    print("Article content retrieved successfully!")
    tagged_sentences = pos_tag(tokenize_text(articles))
    entities = recognize_ne(tagged_sentences)
    print(entities[0])
    