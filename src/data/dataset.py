import random
import pickle
import json
import random
from collections import UserDict, OrderedDict
from typing import Dict, Set
from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

from . import User, Representation


class DataSet(UserDict):
    class Parameter:
        def __init__(
            self, user_params: User.Parameter, num_user: int, splits: Dict[str, float]
        ):
            self.user_params = user_params
            self.num_user = num_user
            self.splits = OrderedDict(splits)

        def generate(self, interests: Dict[str, Set["Article"]], seed: int = None):
            """ Generates a dataset from the specified parameters. """

            # Set seed, if specified
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)

            # Generate all the users
            users = [self.user_params.generate(interests) for _ in range(self.num_user)]

            # Create the names for the columns
            column_names = (
                ["interest_{}".format(i) for i in range(self.user_params.num_interests)]
                + [
                    "article_{}".format(i)
                    for i in range(
                        self.user_params.representation_params.num_articles_per_interest
                        * self.user_params.num_interests
                    )
                ]
                + ["candidate", "label"]
            )

            # Generate the actual dataset
            data = pd.DataFrame.from_records(
                (
                    [interest for interest in interests]
                    + [article for article in articles]
                    + [candidate]
                    + [label]
                    for user in users
                    for interests, articles, candidate, label in user.rows()
                ),
                columns=column_names,
            )

            # Generate the data in a stratified way
            complete_length = len(data)
            results = OrderedDict()
            remaining_data = data
            for i, (key, value) in enumerate(self.splits.items()):
                if i + 1 == len(self.splits):
                    results[key] = remaining_data
                    break
                remaining_data, results[key] = train_test_split(
                    remaining_data,
                    test_size=int(value * complete_length),
                    stratify=remaining_data["label"],
                )

            # Reset indeces and print overview
            for key, value in results.items():
                value.reset_index(drop=True, inplace=True)
                print("{} ({} elements):\n{}\n".format(key, len(value), value['label'].value_counts()))

            return DataSet(dataframes=results, hyperparameters=self)

        @staticmethod
        def from_json(path: str) -> "Parameter":
            """ Load the parameter from a file. """

            with open(path, "r") as json_file:
                data = json.load(json_file)

                user_params = data["user"]
                representation_params = user_params["representation"]

                return DataSet.Parameter(
                    user_params=User.Parameter(
                        representation_params=Representation.Parameter(
                            num_articles_per_interest=representation_params[
                                "articles_per_interest"
                            ],
                            num_positive_samples=representation_params[
                                "positive_samples"
                            ],
                            num_negative_samples=representation_params[
                                "negative_samples"
                            ],
                        ),
                        num_interests=user_params["interest"],
                        num_representations=user_params["representations"],
                    ),
                    num_user=data["users"],
                    splits=data["splits"],
                )

    DATASET_NAME = "Dataset_{}"

    def __init__(
        self, dataframes: Dict[str, pd.DataFrame], hyperparameters: "DataSet.Parameter"
    ):
        super().__init__(dataframes)
        self.params = hyperparameters

    def save(self, path: Path, filename: str = "dataset.pickle") -> Path:
        """ Save the dataset in a corresponding folder structure. """

        num_existing_datasets = len([True for f in path.iterdir() if f.is_dir()])

        current_storage = path / DataSet.DATASET_NAME.format(num_existing_datasets)
        current_storage.mkdir()

        with open(current_storage / filename, "wb") as file:
            pickle.dump(self, file)
        return current_storage

    @staticmethod
    def get_last_dataset(path: Path) -> Path:
        """ Returns the most recently generated dataset. """

        candidates = {
            int(f.name.split("_")[1]): f for f in path.iterdir() if f.is_dir()
        }
        return candidates[max(candidates.keys())]

    @staticmethod
    def load(
        path: Path, version: int = None, filename: str = "dataset.pickle"
    ) -> "DataSet":
        """ Load the dataset from a specified path. """

        dataset_path = (
            DataSet.get_last_dataset(path)
            if version is None
            else path / DataSet.DATASET_NAME.format(version)
        )
        with open(dataset_path / filename, "rb") as file:
            return pickle.load(file)
