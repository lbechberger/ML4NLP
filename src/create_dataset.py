from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from typing import List, Set, Dict
import pickle

from modules.knowledgestore import ks
from modules.wikinews import Article, Category


def articles_to_categories(articles: List[Article]) -> Dict["str", Set[Article]]:
    """ Maps the articles to a list of categories. """
    
    categories = defaultdict(set)
    for article in articles:
        for category in article.categories:
            categories[category].add(article)
    return categories


def create_article(link: str) -> Article:
    return Article(link[22:])


def load_categories(
    cache_file: Path, min_articles: int, num_threads: int = 6
) -> Dict["str", Set[Article]]:
    """ Load the data either from cache or online. """

    if not cache_file.exists():
        print("Generating articles for KnowledgeStore")
        with Pool(num_threads) as pool:
            articles = pool.map(create_article, ks.get_all_resource_uris())
        with open(cache_file, "wb") as file:
            pickle.dump(articles, file)
    else:
        with open(cache_file, "rb") as file:
            articles = pickle.load(file)

    categories = articles_to_categories(articles)
    return Category.filter_categories(categories, min_articles)
