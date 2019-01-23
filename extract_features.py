import json
import shelve
import sys
from pathlib import Path
from typing import Sequence, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append("..")
from src import PATH_DATASETS, PATH_CONFIG
from modules.data import DataSet
from modules.features import FeatureExtractor


def get_unique_articles(dataset: pd.DataFrame):
    article_columns = [
        column for column in dataset.columns.values if column.startswith("article_")
    ]
    return pd.unique(dataset[article_columns + ["candidate"]].values.ravel("K"))


def extract_features(
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
        (length, sum(extractor.get_num_features() for extractor in feature_extractors)),
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
            feature_vectors[
                representation_id,
                feature_index : feature_index + extractor.get_num_features(),
            ] = extractor(
                [(i, articles_parsed[i]) for i in ids_articles],
                (id_candidate, articles_parsed[id_candidate]),
            )

            feature_index += extractor.get_num_features()

    return feature_vectors, labels


def load_feature_extractors(file_name="features.json"):
    with open(PATH_CONFIG / file_name, "r") as json_file:
        data = json.load(json_file)

    return [
        FeatureExtractor.from_name(extractor["name"], extractor.get("arguments", None))
        for extractor in data["extractors"]
    ]


def get_features(
    cache_name: str = "cache.de"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file = Path(cache_name)

    if not file.exists():
        dataset = DataSet.load(PATH_DATASETS)
        feature_extractors = load_feature_extractors()
        nlp = spacy.load("en_core_web_lg")

        X_train, y_train = extract_features(
            nlp, dataset["training"], feature_extractors
        )
        X_test, y_test = extract_features(nlp, dataset["testing"], feature_extractors)

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        split = np.zeros((len(X)), dtype=np.int8)
        split[: len(X_train)] = -1

        with shelve.open(file) as shelf:
            shelf["features"] = X
            shelf["labels"] = y
            shelf["split"] = split
    else:
        with shelve.open(file, "r") as shelf:
            X = shelf["features"]
            y = shelf["labels"]
            split = shelf["split"]

    return X, y, split


if __name__ == "__main__":
    NUM_FEATURES = [5, 10, 15, 20]

    pipeline = Pipeline(
        [("preprocessing", None), ("reduce_dim", None), ("classifier", None)]
    )

    param_grid = [
        {"preprocessing": [None, StandardScaler()]},
        {
            "reduce_dim": [PCA(iterated_power=7), NMF()],
            "reduce_dim__n_components": NUM_FEATURES,
        },
        {"reduce_dim": [SelectKBest(chi2)], "reduce_dim__k": NUM_FEATURES},
        {
            "classifier": [RandomForestClassifier(), ExtraTreesClassifier()],
            "classifier_n_estimators": [5, 25, 50],
        },
        {
            "classifier": [
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                KNeighborsClassifier(3),
                GaussianProcessClassifier(),
                SVC()
            ]
        },
    ]

    X, y, split = get_features()
    grid = GridSearchCV(
        pipeline,
        cv=PredefinedSplit(split),
        n_jobs=-1,
        param_grid=param_grid,
        scoring="f1",
    )
    grid.fit(X, y)

    result = pd.DataFrame(grid.cv_results_)
