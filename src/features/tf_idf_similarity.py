from typing import Sequence, List, Tuple, Any

from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

from src.features import FeatureExtractor


class TfIdfSimilarity(FeatureExtractor):
    """
    Compare the similarity of articles given their Tf-Idf vectors.
    Unlike the "LocalSimilarity", all existing articles of the training set are considered.
    """

    def __init__(self, num_features: int = 1000):
        """
        Create the feature extractor.
        :param features: The number of features used by the underlying TF-IDF engine.
        """

        self.__matrix = None
        self.features = num_features

    def prepare(self, unique_articles: Sequence[Doc]) -> None:
        vectorizer = TfidfVectorizer(
            lowercase=False, max_features=self.features, dtype=np.float32
        )

        # Calculate the Tf-Id matrix
        self.__matrix = vectorizer.fit_transform(
            (
                " ".join(
                    token.lemma_
                    for token in parsed_article
                    if not token.is_stop and not token.is_punct
                )
                for parsed_article in unique_articles
            )
        )

    def __call__(
        self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]
    ) -> Sequence[float]:
        # Calculate the cosine distance between the vectors if the articles and the candidate.
        candidate_tfidf = self.__matrix[candidate[0]].todense()
        tfidf_similarities = [
            cosine(self.__matrix[i].todense(), candidate_tfidf) for i, _ in articles
        ]

        return TfIdfSimilarity.aggregate(tfidf_similarities)
