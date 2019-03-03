from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

from src.features import FeatureExtractor


class LocalSimilarity(FeatureExtractor):
    def __init__(self, num_features=100, weighed=True):
        self.num_features = num_features
        self.weighted = weighed

    def __call__(self, articles, candidate):
        lemmatized_articles = [
            " ".join(
                token.lemma_
                for token in article
                if not token.is_stop and not token.is_punct
            )
            for _, article in chain([candidate], articles)
        ]

        vectorizer = TfidfVectorizer(
            lowercase=False, max_features=self.num_features, dtype=np.float32
        )
        tf_idf_matrix = vectorizer.fit_transform(lemmatized_articles)

        candidate_tfidf = tf_idf_matrix[0].todense()
        tfidf_similarities = [
            cosine(tf_idf_matrix[i].todense(), candidate_tfidf)
            for i in range(1, len(articles) + 1)
        ]

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

        vector_similarities = [
            cosine(mean_vectors[i], mean_vectors[0])
            for i in range(1, len(mean_vectors))
        ]

        return list(FeatureExtractor.aggregate(tfidf_similarities)) + list(
            FeatureExtractor.aggregate(vector_similarities)
        )

    @classmethod
    def get_num_features(cls):
        return 2 * super().get_num_features()
