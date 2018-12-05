import random
import pickle
import json
import random
from collections import UserDict

import pandas as pd
from sklearn.utils import shuffle
import numpy as np

from . import User, Representation
from pathlib import Path


class DataSet(UserDict):
    class Parameter:
        def __init__(self, user_params, num_user, splits):
            self.user_params = user_params
            self.num_user = num_user
            self.splits = splits

        def generate(self, interests, seed=None):
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

            # Shuffle the data
            data = shuffle(data)

            # Generate the splits
            index = 0
            result = {}
            for key, value in self.splits.items():
                size = int(len(data) * value)
                result[key] = data.iloc[
                    index : int(index + len(data) * value)
                ].reset_index(drop=True)
                index += size

            return DataSet(dataframes=result, hyperparameters=self)

        @staticmethod
        def from_json(path):
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

    def __init__(self, dataframes, hyperparameters):
        super().__init__(dataframes)
        self.params = hyperparameters

    def save(self, path, filename="dataset.pickle"):
        num_existing_datasets = len([True for f in path.iterdir() if f.is_dir()])

        current_storage = path / DataSet.DATASET_NAME.format(num_existing_datasets)
        current_storage.mkdir()

        with open(current_storage / filename, "wb") as file:
            pickle.dump(self, file)
        return current_storage

    @staticmethod
    def get_last_dataset(path):
        candidates = {
            int(f.name.split("_")[1]): f for f in path.iterdir() if f.is_dir()
        }
        return candidates[max(candidates.keys())]

    @staticmethod
    def load(path, version=None, filename="dataset.pickle"):
        dataset_path = (
            DataSet.get_last_dataset(path)
            if version is None
            else path / DataSet.DATASET_NAME.format(version)
        )
        with open(dataset_path / filename, "rb") as file:
            return pickle.load(file)
