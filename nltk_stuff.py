import nltk

text = 'John Wilkes Booth shot Abraham Lincoln. This did not happen inside the White House'
sentences = nltk.sent_tokenize(text)
for sent in sentences:
    word_tokenized = nltk.word_tokenize(sent)
    pos_tagged = nltk.pos_tag(word_tokenized)
    ne_chunked = nltk.ne_chunk(pos_tagged)
    print(ne_chunked)