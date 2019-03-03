from src.features import FeatureExtractor


class DocumentSimilarity(FeatureExtractor):
    def __call__(self, articles, candidate):
        return DocumentSimilarity.aggregate(
            [article.similarity(candidate[1]) for _, article in articles]
        )
