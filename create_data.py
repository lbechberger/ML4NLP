from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from typing import List, Set, Dict
import pickle

from matplotlib import pyplot
import seaborn as sbs
import numpy as np
from wordcloud import WordCloud

from src import PATH_CONFIG, PATH_RAW_DATA, PATH_DATASETS
from src.data import DataSet
from src.wikinews import Article, Category
from src.knowledgestore import ks

SEED = 42
MIN_ARTICLES = 5


def _create_article(link: str) -> Article:
    return Article(link[22:])


def load_categories(
    cache_file: Path, min_articles: int, num_threads: int = 6
) -> Dict["str", Set[Article]]:
    def articles_to_categories(articles: List[Article]) -> Dict["str", Set[Article]]:
        """ Maps the articles to a list of categories. """

        categories = defaultdict(set)
        for article in articles:
            for category in article.categories:
                categories[category].add(article)
        return categories

    if not cache_file.exists():
        print("Loading URLs...")
        urls = ks.get_all_resource_uris()
        print(
            "Generating articles for KnowledgeStore from {} urls...".format(len(urls))
        )
        if num_threads > 1:
            with Pool(num_threads) as pool:
                articles = pool.map(_create_article, urls)
        else:
            articles = [_create_article(url) for url in urls]

        with open(cache_file, "wb") as file:
            pickle.dump(articles, file)
    else:
        with open(cache_file, "rb") as file:
            articles = pickle.load(file)

    articles = [x for x in articles if len(x.text) > 0]
    categories = articles_to_categories(articles)
    return Category.filter_categories(categories, min_articles)


def visualize(interests):
    def visualize_distibution(interests):
        num_articles = np.array([category[1] for category in interests])
        fig, ax = pyplot.subplots(figsize=(11.7, 8.27))
        dist = sbs.distplot(num_articles, hist=False, norm_hist=False, ax=ax)
        dist.set_xlabel("Number of articles per category (log)")
        dist.set_ylabel("Relative number of categories")
        dist.set_xscale("log")
        dist.set_title("Distibution of articles")
        dist.set_xlim(0)

    def visualize_words(interests):
        top_occurences = {name: occurences for name, occurences in interests}
        sum_occurences = sum(top_occurences.values())
        top_occurences = {
            name: occurences / sum_occurences
            for name, occurences in top_occurences.items()
        }

        cloud = WordCloud(background_color="white").fit_words(top_occurences)

        fig, ax = pyplot.subplots(figsize=(11.7, 8.27))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")

    interests_tmp = [(key, len(value)) for key, value in interests.items()]
    interests_tmp.sort(key=lambda x: x[1], reverse=True)

    visualize_distibution(interests_tmp)
    visualize_words(interests_tmp)


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

    # Visualize the interests
    visualize(interests)

    # Load the dataset parameter from the config file and generate the data with a specific seed
    data = DataSet.Parameter.from_json(PATH_CONFIG / "dataset.json").generate(
        interests, SEED
    )

    # Save the data and export it into human-readable *.csv
    data_path = data.save(PATH_DATASETS)
    for dataset_name, dataset in data.items():
        dataset.to_csv(data_path / "{}.csv".format(dataset_name), sep="\t", index=False)
