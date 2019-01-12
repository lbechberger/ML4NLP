import sys
from typing import Sequence, Optional

import spacy
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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


if __name__ == "__main__":
    dataset = DataSet.load(PATH_DATASETS)
    feature_extractors = load_feature_extractors()

    nlp = spacy.load("en_core_web_lg")

    X_train, y_train = extract_features(nlp, dataset["training"], feature_extractors)
    X_test, y_test = extract_features(nlp, dataset["testing"], feature_extractors)

    #%%
    pipeline = Pipeline([("rf", RandomForestClassifier(n_estimators=50))])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    print("F1-score: ", f1_score(y_true=y_test, y_pred=prediction))
    print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=prediction))
    print("Precision: ", precision_score(y_true=y_test, y_pred=prediction))
    print(confusion_matrix(y_true=y_test, y_pred=prediction))
