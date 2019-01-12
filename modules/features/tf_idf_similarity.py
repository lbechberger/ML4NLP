from statistics import mean, median

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

from modules.features import FeatureExtractor


class TfIdfSimilarity(FeatureExtractor):
    def prepare(self, unique_articles):
        vectorizer = TfidfVectorizer(
            lowercase=False, max_features=1000, dtype=np.float32
        )
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

    def __call__(self, articles, candidate):
        candidate_tfidf = self.__matrix[candidate[0]].todense()
        tfidf_similarities = [
            cosine(self.__matrix[i].todense(), candidate_tfidf) for i, _ in articles
        ]

        return (
            mean(tfidf_similarities),
            median(tfidf_similarities),
            min(tfidf_similarities),
            max(tfidf_similarities),
        )
