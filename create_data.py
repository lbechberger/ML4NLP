from pathlib import Path

from modules.data import DataSet
from src.create_dataset import load_categories

DATASET_DIR = Path("datasets")
CONFIG_DIR = Path("config")
RAW_DATA = Path("raw_data")

if __name__ == "__main__":
    assert CONFIG_DIR.exists() and RAW_DATA.exists(), "Invalid paths!"

    if not DATASET_DIR.exists():
        DATASET_DIR.mkdir()

    existing_datasets = len([f for f in DATASET_DIR.iterdir() if f.is_dir()])
    current_storage = Path(DATASET_DIR, "Dataset{}".format(existing_datasets))
    current_storage.mkdir()

    interests = load_categories(RAW_DATA / "articles.pickle", 5)

    dataset = DataSet.Parameter.from_json(CONFIG_DIR / "dataset.json").generate(interests)
    dataset.save(current_storage / "dataset.pickle")
    dataset.as_dataframe().to_csv(current_storage / "dataset.csv", sep="\t")
