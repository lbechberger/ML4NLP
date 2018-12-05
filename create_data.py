from pathlib import Path

from modules.data import DataSet
from src.create_dataset import load_categories
from src import PATH_CONFIG, PATH_RAW_DATA, PATH_DATASETS

SEED = 42
MIN_ARTICLES = 5

if __name__ == "__main__":
    assert PATH_CONFIG.exists(), "Invalid config path!"

    # Create raw data cache, if needed
    if not PATH_RAW_DATA.exists():
        PATH_RAW_DATA.mkdir()

    # Create dataset dir, if needed
    if not PATH_DATASETS.exists():
        PATH_DATASETS.mkdir()

    # Load all available categories from cache or web and filter for those with at least MIN_ARTICLES
    interests = load_categories(PATH_RAW_DATA / "articles.pickle", MIN_ARTICLES)

    # Load the dataset parameter from the config file and generate the data with a specific seed
    data = DataSet.Parameter.from_json(PATH_CONFIG / "dataset.json").generate(
        interests, SEED
    )

    # Save the data and export it into human-readable *.csv
    data_path = data.save(PATH_DATASETS)
    for dataset_name, dataset in data.items():
        dataset.to_csv(data_path / "{}.csv".format(dataset_name), sep="\t", index=False)
