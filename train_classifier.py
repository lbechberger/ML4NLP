import sys
import itertools
import copy
from collections import OrderedDict
import json
import shelve
import sys
from typing import Sequence, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

sys.path.append("..")
from src import PATH_DATASETS, PATH_CONFIG
from src.data import DataSet
from src.features import FeatureExtractor


class Features:
    def __init__(self, feature_cache: Union[Path, str]):
        self.__cache_file = feature_cache

    def __bool__(self):
        with shelve.open(self.__cache_file, "c") as shelf:
            return "features" in shelf

    def load(
        self, dataset_path: Path, feature_config_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with shelve.open(self.__cache_file, "c") as shelf:
            if "features" not in shelf:
                print("[INFO] Extracting features")
                dataset = DataSet.load(dataset_path)
                feature_extractors = Features.__load_feature_extractors(
                    feature_config_path
                )
                nlp = spacy.load("en_core_web_lg")

                X_train, y_train = Features.__extract_features(
                    nlp, dataset["training"], feature_extractors
                )
                X_validation, y_validation = Features.__extract_features(
                    nlp, dataset["validation"], feature_extractors
                )
                X_test, y_test = Features.__extract_features(
                    nlp, dataset["testing"], feature_extractors
                )

                X = np.concatenate((X_train, X_validation))
                y = np.concatenate((y_train, y_validation))
                split = np.zeros((len(X)), dtype=np.int8)
                split[: len(X_train)] = -1

                shelf["features"] = X
                shelf["labels"] = y
                shelf["validation_split"] = split
                shelf["test_features"] = X_test
                shelf["test_labels"] = y_test

            return (
                shelf["features"],
                shelf["labels"],
                shelf["validation_split"],
                shelf["test_features"],
                shelf["test_labels"],
            )

    @staticmethod
    def __extract_features(
        nlp_parser,
        dataset: pd.DataFrame,
        feature_extractors: Sequence[FeatureExtractor],
        length: Optional[int] = None,
    ):
        # Extract columns of articles and unique articles in a dataset
        article_columns = [
            column for column in dataset.columns.values if column.startswith("article_")
        ]
        unique_articles = pd.unique(
            dataset[article_columns + ["candidate"]].values.ravel("K")
        )
        assert all(
            len(article.text) > 0 for article in unique_articles
        ), "Articles without text found!"

        # Parse all texts, create a mapping and prepare the feature extractors
        articles_parsed = list(
            nlp_parser.pipe((article.text for article in unique_articles))
        )
        articles_mapping = {article.url: i for i, article in enumerate(unique_articles)}
        for feature in feature_extractors:
            feature.prepare(articles_parsed)

        # Create the required outputs
        length = length if length is not None else len(dataset)
        data = dataset.iloc[:length]
        labels = [x for x in dataset["label"][:length]]
        feature_vectors = np.zeros(
            (
                length,
                sum(extractor.get_num_features() for extractor in feature_extractors),
            ),
            dtype=np.float32,
        )

        for representation_id, representation in data.iterrows():
            id_candidate = articles_mapping[representation["candidate"].url]
            ids_articles = [
                articles_mapping[value.url]
                for key, value in representation.iteritems()
                if key in article_columns
            ]

            # Fill the feature vector
            feature_index = 0
            for extractor_id, extractor in enumerate(feature_extractors):
                num_features = extractor.get_num_features()
                feature_vectors[
                    representation_id, feature_index : (feature_index + num_features)
                ] = extractor(
                    [(i, articles_parsed[i]) for i in ids_articles],
                    (id_candidate, articles_parsed[id_candidate]),
                )
                feature_index += num_features

        return feature_vectors, labels

    @staticmethod
    def __load_feature_extractors(file_name: Path):
        with open(str(file_name), "r") as json_file:
            data = json.load(json_file)

        return [
            FeatureExtractor.from_name(
                extractor["name"], extractor.get("arguments", None)
            )
            for extractor in data["extractors"]
        ]


def _generate_transformer_versions(transformer, **kwargs):
    """
    This function generates all possible permutations of a tranformer with mutliple arguments.
    :param transformer: The class of the transformer
    :param kwargs: The arguments of the transformer. Sequence are interpreted as different versions.
    :return: Generator of Transformer instances
    """

    fixed_arguments = dict(
        (name, argument)
        for name, argument in kwargs.items()
        if not isinstance(argument, Sequence)
    )

    free_arguments = OrderedDict(
        (name, argument)
        for name, argument in kwargs.items()
        if isinstance(argument, Sequence)
    )

    for combination in itertools.product(*free_arguments.values()):
        arguments = copy.deepcopy(fixed_arguments)
        for argument_name, value in zip(free_arguments.keys(), combination):
            arguments[argument_name] = value
        yield transformer(**arguments)


if __name__ == "__main__":
    NUM_FEATURES = [5, 10, 15, 20]

    pipeline = Pipeline(
        [("preprocessing", None), ("reduce_dim", None), ("classifier", None)]
    )

    param_grid = {
        "preprocessing": [None, StandardScaler()],
        "reduce_dim": list(
            itertools.chain(
                [None],
                _generate_transformer_versions(
                    PCA, iterated_power=7, n_components=NUM_FEATURES
                ),
            )
        ),
        "classifier": list(
            itertools.chain(
                _generate_transformer_versions(
                    RandomForestClassifier, n_estimators=[8, 16, 32, 64, 128]
                ),
                _generate_transformer_versions(
                    ExtraTreesClassifier, n_estimators=[8, 16, 32, 64, 128]
                ),
                [
                    GaussianNB(),
                    QuadraticDiscriminantAnalysis(),
                    KNeighborsClassifier(3),
                    SVC(gamma="auto"),
                ],
            )
        ),
    }

    # Generate or load the features
    features = Features("features")
    X, y, split, X_test, y_test = features.load(
        dataset_path=PATH_DATASETS, feature_config_path=PATH_CONFIG / "features.json"
    )

    # Do the grid search
    grid = GridSearchCV(
        pipeline,
        cv=PredefinedSplit(split),
        n_jobs=-1,
        param_grid=param_grid,
        scoring="f1",
        error_score="raise",
        return_train_score=True,
    )
    grid.fit(X, y)

    # Output the results
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values("rank_test_score", inplace=True, ascending=True)
    results.to_csv("results.csv", "\t", index=False)

    print("Results on test set:")
    print(confusion_matrix(y_test, grid.predict(X_test)))
