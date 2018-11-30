import pandas as pd
import shelve
import knowledgestore.ks as ks

class ArticleCache():

    def __init__(self, all_articles_csv='all_article_uris.csv', cache_file='article_cache.shelve'):
        self.all_articles_csv = all_articles_csv
        self.cache_file = cache_file
        self.articles = pd.read_csv(all_articles_csv).article


    def generate_articles(self, upto=0, verbose=False):
        articles = self.articles[:upto] if upto else self.articles
        with shelve.open(self.cache_file) as cache:
            for num, uri in enumerate(articles):
                if verbose:
                    print('Generating Article #'+str(num))
                if uri not in cache:
                    try:
                        cache[uri] = ks.run_files_query(uri)
                    except ConnectionError:
                        continue
                yield cache[uri]


    def cache_all_articles(self):
        with shelve.open(self.cache_file) as cache:
            num = len(cache)
            for uri in self.articles[num:]:
                num += 1
                if num % 200 == 0:
                    print("Caching #"+str(num)+'...')
                cache[uri] = ks.run_files_query(uri)


    def get_article(self, uri):
        with shelve.open(self.cache_file, 'r') as cache:
            if uri not in cache:
                return ks.run_files_query(uri)
            return cache[uri]

    def __getitem__(self, item):
        return self.get_article(item)