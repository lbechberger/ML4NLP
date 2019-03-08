from typing import Sequence, List, Tuple, Any

from src.features import FeatureExtractor


class DocumentSimilarity(FeatureExtractor):
    """
    Calculate the article similarity utilizing the CNN-based similarity metric provided by SpaCy.
    """

    def __call__(
        self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]
    ) -> Sequence[float]:
        return DocumentSimilarity.aggregate(
            [article.similarity(candidate[1]) for _, article in articles]
        )
