import pickle
import json
import random
from collections import UserDict, OrderedDict
from typing import Dict, Set, Optional
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from . import User, Representation


class DataSet(UserDict):
    """
    A dict-like structure containing the different (i.e. training & testing) datasets as DateFrame.
    """

    class Parameter:
        """
        The parameters to create a DataSet.
        """

        def __init__(
            self, user_params: User.Parameter, num_user: int, splits: Dict[str, float]
        ):
            """
            Create a new object with the parameters for a DataSet.
            :param user_params: The parameters for the included users.
            :param num_user: The number of users included in the datasets.
            :param splits: The different datasets. Must sum up to 1.
            """

            assert sum(splits.values()) == 1.0, "The splits must sum up to 1.0"
            self.user_params = user_params
            self.num_user = num_user
            self.splits = OrderedDict(splits)

        def generate(
            self, interests: Dict[str, Set["Article"]], seed: int = None
        ) -> "DataSet":
            """
            Generates a dataset from the specified parameters.
            :param interests: The available interest a user may have.
            :param seed: A seed for pseudo-random generator
            :return: The dataset of interest.
            """

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
                    for interests, articles, candidate, label in user.samples()
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
                print(
                    "{} ({} elements):\n{}\n".format(
                        key, len(value), value["label"].value_counts()
                    )
                )

            return DataSet(dataframes=results, hyperparameters=self)

        @staticmethod
        def from_json(path: str) -> "Parameter":
            """
            Load the parameter from a file.
            :param path: The path to the JSON file.
            :return: A Parameter object.
            """

            with open(path, "r") as json_file:
                # Parse the included JSON in the file.
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
        """
        Create a dataset. You should use "DataSet.Parameter.generate"!
        :param dataframes: Different datasets.
        :param hyperparameters: The parameters used to generate the datasets.
        """

        super().__init__(dataframes)
        self.params = hyperparameters

    def save(self, path: Path, filename: str = "dataset.pickle") -> Path:
        """
        Save the dataset in a corresponding folder structure.
        :param path: A path containing the different generated datasets.
        :param filename: The filename for the serialized dataset.
        :return: The path to the folder with the serialized dataset.
        """

        num_existing_datasets = len([True for f in path.iterdir() if f.is_dir()])

        current_storage = path / DataSet.DATASET_NAME.format(num_existing_datasets)
        current_storage.mkdir()

        with open(str(current_storage / filename), "wb") as file:
            pickle.dump(self, file)
        return current_storage

    @staticmethod
    def get_last_dataset(path: Path) -> Path:
        """
        Returns the path to the most recently generated dataset.
        :param path: A path containing the different generated datasets.
        :return: The path to the most recently generated dataset.
        """

        candidates = {
            int(f.name.split("_")[1]): f for f in path.iterdir() if f.is_dir()
        }
        return candidates[max(candidates.keys())]

    @staticmethod
    def load(
        path: Path, version: Optional[int] = None, filename: str = "dataset.pickle"
    ) -> "DataSet":
        """
        Load the dataset from a specified path.
        :param path: The path to the folder with the datasets.
        :param version: The number of the target dataset. If None, choose the most recent one.
        :param filename: The filename for the serialized dataset.
        :return: Loaded dataset from the disk.
        """

        dataset_path = (
            DataSet.get_last_dataset(path)
            if version is None
            else path / DataSet.DATASET_NAME.format(version)
        )
        with open(str(dataset_path / filename), "rb") as file:
            return pickle.load(file)
