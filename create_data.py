from pathlib import Path

from modules.data import DataSet
from src.create_dataset import load_categories
from src import PATH_CONFIG, PATH_RAW_DATA, PATH_DATASETS

if __name__ == "__main__":
    assert PATH_CONFIG.exists(), "Invalid config path!"

    if not PATH_RAW_DATA.exists():
        PATH_RAW_DATA.mkdir()

    if not PATH_DATASETS.exists():
        PATH_DATASETS.mkdir()

    interests = load_categories(PATH_RAW_DATA / "articles.pickle", 5)

    dataset = DataSet.Parameter.from_json(PATH_CONFIG / "dataset.json").generate(interests)
    dataset_path = dataset.save(PATH_DATASETS)
    dataset.as_dataframe().to_csv(dataset_path / "dataset.csv", sep="\t")
