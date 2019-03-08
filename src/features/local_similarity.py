from typing import Sequence, List, Tuple, Any
from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

from src.features import FeatureExtractor


class LocalSimilarity(FeatureExtractor):
    """
    Compare the similarity of articles given their local Tf-Idf vectors and word vectors.
    """

    def __init__(self, num_features: int = 100, weighed: bool = True):
        """
        Create a new feature extractor.
        :param num_features: The number of features used by the underlying TF-IDF engine.
        :param weighed: Weight the word vectors by their Tf-Idf scores.
        """

        self.num_features = num_features
        self.weighted = weighed

    def __call__(
        self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]
    ) -> Sequence[float]:

        # Create a lemmatized version of the articles without stiop words and punctation.
        lemmatized_articles = [
            " ".join(
                token.lemma_
                for token in article
                if not token.is_stop and not token.is_punct
            )
            for _, article in chain([candidate], articles)
        ]

        # Learn a Tf-Idf score utilizing the lemmatized articles.
        vectorizer = TfidfVectorizer(
            lowercase=False, max_features=self.num_features, dtype=np.float32
        )
        tf_idf_matrix = vectorizer.fit_transform(lemmatized_articles)

        # Calculate the cosine distance between the Tf-Idf vector of the articles and the candidate.
        candidate_tfidf = tf_idf_matrix[0].todense()
        tfidf_similarities = [
            cosine(tf_idf_matrix[i].todense(), candidate_tfidf)
            for i in range(1, len(articles) + 1)
        ]

        # Calculate the (weighted) mean vectors of the word vectors
        mean_vectors = [
            np.mean(
                np.stack(
                    [
                        token.vector
                        * (
                            tf_idf_matrix[i, vectorizer.vocabulary_[token.lemma_]]
                            if self.weighted
                            else 1.0
                        )
                        for token in article
                        if token.lemma_ in vectorizer.vocabulary_
                    ]
                )
            )
            for i, (_, article) in enumerate(chain([candidate], articles))
        ]

        # Calculate the cosine distance between the mean vectors of the articles and the candidate.
        vector_similarities = [
            cosine(mean_vectors[i], mean_vectors[0])
            for i in range(1, len(mean_vectors))
        ]

        # Create a fixed-length number of features
        return list(FeatureExtractor.aggregate(tfidf_similarities)) + list(
            FeatureExtractor.aggregate(vector_similarities)
        )

    @classmethod
    def get_num_features(cls):
        return 2 * super().get_num_features()
