from statistics import mean, median

from modules.features import FeatureExtractor


class DocumentSimilarity(FeatureExtractor):
    def __call__(self, articles, candidate):
        cosine_diff = [article.similarity(candidate[1]) for _, article in articles]
        return (
            mean(cosine_diff),
            median(cosine_diff),
            min(cosine_diff),
            max(cosine_diff),
        )
