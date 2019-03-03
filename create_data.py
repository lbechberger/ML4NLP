from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from typing import List, Set, Dict
import pickle


from src import PATH_CONFIG, PATH_RAW_DATA, PATH_DATASETS
from src.data import DataSet
from src.wikinews import Article, Category
from src.knowledgestore import ks

SEED = 42
MIN_ARTICLES = 5


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
        print("Loading URLs...")
        urls = ks.get_all_resource_uris()
        print(
            "Generating articles for KnowledgeStore from {} urls...".format(len(urls))
        )
        if num_threads > 1:
            with Pool(num_threads) as pool:
                articles = pool.map(create_article, urls)
        else:
            articles = [create_article(url) for url in urls]

        with open(cache_file, "wb") as file:
            pickle.dump(articles, file)
    else:
        with open(cache_file, "rb") as file:
            articles = pickle.load(file)

    articles = [x for x in articles if len(x.text) > 0]
    categories = articles_to_categories(articles)
    return Category.filter_categories(categories, min_articles)


if __name__ == "__main__":
    assert PATH_CONFIG.exists(), "Invalid config path!"

    # Create raw data cache, if needed
    if not PATH_RAW_DATA.exists():
        PATH_RAW_DATA.mkdir()

    # Create dataset dir, if needed
    if not PATH_DATASETS.exists():
        PATH_DATASETS.mkdir()

    # Load all available categories from cache or web and filter for those with at least MIN_ARTICLES
    interests = load_categories(
        PATH_RAW_DATA / "articles.pickle", MIN_ARTICLES, num_threads=4
    )

    # Load the dataset parameter from the config file and generate the data with a specific seed
    data = DataSet.Parameter.from_json(PATH_CONFIG / "dataset.json").generate(
        interests, SEED
    )

    # Save the data and export it into human-readable *.csv
    data_path = data.save(PATH_DATASETS)
    for dataset_name, dataset in data.items():
        dataset.to_csv(data_path / "{}.csv".format(dataset_name), sep="\t", index=False)
