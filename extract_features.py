import sys
from statistics import mean, median

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.append("..")
from src import PATH_RAW_DATA, PATH_DATASETS
from modules.data import DataSet


def calculate_cosine_distance(articles, candidate):
    candidate_vector = candidate.vector.reshape(1, -1)
    cosine_diff = [
        float(cosine_distances(article.vector.reshape(1, -1), candidate_vector)[0, 0])
        for article in articles
    ]
    return mean(cosine_diff), median(cosine_diff), min(cosine_diff), max(cosine_diff)


def calculate_entity_iou(articles, candidate, label):
    def extract_entities(text, label):
        return frozenset(
            entity.lemma_.split()[-1]
            for entity in text.ents
            if entity.label_ == label and len(entity.lemma_) > 0
        )

    candidate_entities = extract_entities(candidate, label)
    ious = []
    for article in articles:
        entities = extract_entities(article, label)
        union = len(candidate_entities.union(entities))
        ious.append(
            len(candidate_entities.intersection(entities)) / union if union > 0 else 0
        )

    return mean(ious), median(ious), min(ious), max(ious)


def extract_features(dataset, feature_extractors, length=None):
    length = length if length is not None else len(dataset)
    feature_vectors = np.zeros((length, len(feature_extractors) * 4), dtype=np.float32)
    data = dataset.iloc[:length]
    labels = [x for x in dataset["label"][:length]]

    for representation_id, representation in data.iterrows():
        candidate = representation["candidate"].parsed_text(nlp)
        articles = [
            value.parsed_text(nlp)
            for key, value in representation.iteritems()
            if key.startswith("article_")
        ]

        for extractor_id, extractor in enumerate(feature_extractors):
            extractor_id = extractor_id * 4
            feature_vectors[
                representation_id, extractor_id : extractor_id + 4
            ] = extractor(articles, candidate)

    return feature_vectors, labels


if __name__ == "__main__":
    dataset = DataSet.load(PATH_DATASETS)
    nlp = spacy.load("en_core_web_lg")

    #%%
    LENGTH = len(dataset["training"])
    FEATURE_EXTRACTORS = [
        calculate_cosine_distance,
        lambda articles, candidate: calculate_entity_iou(articles, candidate, "PERSON"),
        lambda articles, candidate: calculate_entity_iou(articles, candidate, "NORP"),
        lambda articles, candidate: calculate_entity_iou(articles, candidate, "ORG"),
        lambda articles, candidate: calculate_entity_iou(articles, candidate, "GPE"),
        lambda articles, candidate: calculate_entity_iou(articles, candidate, "EVENT"),
    ]

    X_train, y_train = extract_features(dataset["training"], FEATURE_EXTRACTORS)
    X_test, y_test = extract_features(dataset["testing"], FEATURE_EXTRACTORS)

    #%%
    pipeline = Pipeline([("rf", RandomForestClassifier(n_estimators=50))])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    print("F1-score: ", f1_score(y_true=y_test, y_pred=prediction))
    print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=prediction))
    print("Precision: ", precision_score(y_true=y_test, y_pred=prediction))
    print(confusion_matrix(y_true=y_test, y_pred=prediction))
