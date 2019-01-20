from generation_functions import *
from sklearn.metrics import pairwise
import gensim, os


class FeatureExtraction:

    def __init__(self, articles, categories):
        self.articles = articles
        self.categories = categories

    def get_category_embeddings(self, embedding=[]):
        """
        Calculate the embedding vectors of the categories
        :param embedding: embedding vector. if not provided, google's word2vec is used
        :return: mean embedding of each category name
        """
        # Embedding of each category
        if os.path.isfile('./data/embed_categories.pickle'):
            embedded_categories = pickle.load(open("./data/embed_categories.pickle", "rb"))
        else:
            if not embedding:
                embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                            binary=True).wv
            # calculate embeddings and take mean
            embedded_categories = []
            for c in self.categories:
                if c == "Wackynews":
                    c = "Wacky News"
                # get the embeddings
                embeds = get_w2v_string(c, embedding)
                # calculate mean if there was more than one word
                if len(embeds) > 1:
                    embeds = [sum(x)/len(x) for x in zip(*embeds)]
                else:
                    embeds = embeds[0]
                # append embeds to list of all category embeddings
                embedded_categories.append(embeds)

        pickle.dump(embedded_categories, open("./data/embed_categories.pickle", "wb"))

        return embedded_categories

    def get_article_embedding(self, embedding=[]):
        """
        Calculate the embeddings of each article
        :param embedding: embedding vector. if not provided, google's word2vec is used
        :return: mean embedding of each articles
        """
        if os.path.isfile('./data/embed_articles.pickle'):
            embedded_articles = pickle.load(open("./data/embed_articles.pickle", "rb"))
        else:
            if not embedding:
                embedding = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                            binary=True).wv
            a_counter = 1
            embedded_articles = []
            for a in self.articles:
                embed = get_w2v_string(a, embedding)

                # append mean embedding of article to list of all article embeddings
                embedded_articles.append([sum(x)/len(x) for x in zip(*embed)])
                print("Article: {} /{}".format(a_counter, len(self.articles)))
                a_counter += 1

            pickle.dump(embedded_articles, open("./data/embed_articles.pickle", "wb"))
        return embedded_articles

    def get_cosine_sim(self):
        """
        Calculate the cosine similarity between the articles' and categories' embedding vectors
        :return: Float array with the similarity measures
        """
        # Calculate cosine similarity between articles and category names
        cos_sims = []
        embedded_articles = self.get_article_embedding()
        embedded_categories = self.get_category_embeddings()
        for a in embedded_articles:
            cs = []
            for c in embedded_categories:
                cs.append(pairwise.cosine_similarity([a], [c]))
            cos_sims.append(cs)

        return cos_sims

    def extract_user_similarities(self, users):
        """
        Return max, min, mean similarities of the categories the user liked
        :param users: Array of all users where each user's values is in range [0,1]
        :return: Max, min and mean array containing the respective similarities
        """
        if os.path.isfile('./data/cosine_similarities.pickle'):
            max_val, min_val, mean_val = pickle.load(open("./data/cosine_similarities.pickle", "rb"))
            return max_val, min_val, mean_val
        else:
            max_val = []
            min_val = []
            mean_val = []
            similarities = self.get_cosine_sim()
            for idx, user in users.iterrows():
                print(idx)
                for col in range(len(user)):
                    if col == 1:
                        ma = np.max(similarities[col])
                        mi = np.min(similarities[col])
                        me = np.mean(similarities[col])
                        max_val.append(ma)
                        min_val.append(mi)
                        mean_val.append(me)

            pickle.dump([max_val, min_val, mean_val], open("./data/cosine_similarities.pickle", "wb"))
            return max_val, min_val, mean_val

    def get_article_length(self):
        lengths = []
        for a in self.articles:
            lengths.append(len(a))
        return lengths

    def category_check(self):
        """
        Check if the category names appear in each articles
        :return: int array [0,1]
        """
        if os.path.isfile('./data/category_in_articles.pickle'):
            cat_in_articles = pickle.load(open("./data/category_in_articles.pickle", "rb"))
        else:
            cat_in_articles = []
            for a in self.articles:
                a = a.lower()
                bool_array = []
                for name in self.categories:
                    name = name.lower()
                    if name == "wackynews":
                        name = "wacky"  # not news, because news is highly uncorrelated to the category
                    c_in_a = 0
                    for c in name.split(" "):
                        if c == "and":
                            continue
                        if c in a:
                            c_in_a = 1
                            break
                    bool_array.append(c_in_a)
                cat_in_articles.append(bool_array)
            pickle.dump(cat_in_articles, open("./data/category_in_articles.pickle", "wb"))

        return cat_in_articles

    def keyword_check(self, keywords):
        """
        Checks the article for predefined keywords/terms
        :param keywords: txt file with keywords
        :return: array for each categories's keywords in each article[0,1]
        """
        if os.path.isfile('./data/keywords_in_articles.pickle'):
            keys_in_articles = pickle.load(open("./data/keywords_in_articles.pickle", "rb"))
        else:
            keys_in_articles = []
            for a in self.articles:
                a = a.lower()
                l_in_a = []
                for line in keywords:
                    # remove brackets from lines, so they look like: "Term A", "Term B"
                    line = line[line.find("[")+1:line.find("]")]
                    count = 0
                    for w in line.split(", "):  # for each word in line
                        w = w.replace('"', '').lower()  # remove parenthesis from words
                        count += sum(1 for _ in re.finditer(r'%s' % re.escape(w), a))  # count number of appearances
                    l_in_a.append(count)
                keys_in_articles.append(l_in_a)
            pickle.dump(keys_in_articles, open("./data/keywords_in_articles.pickle", "wb"))

        return keys_in_articles
