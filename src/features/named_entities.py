from typing import FrozenSet, Sequence, List, Tuple, Any

from spacy.tokens import Doc

from src.features import FeatureExtractor


class NamedEntities(FeatureExtractor):
    """
    Calculate the similarity beween articles as the intersection-over-union between their named enities.
    """

    def __init__(self, label: str):
        """
        Create a new feature extractor.
        :param label: The label of the named entity.
        """

        self.label = label

    def __call__(
        self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]
    ) -> Sequence[float]:
        def extract_entities(text: Doc, label: str) -> FrozenSet[str]:
            """
            Extract a set of named entities from a document.
            If a lemmatized named entity consists out of multiple words ("angela merkel") just return the last ('merkel')
            :param text: The parsed article.
            :param label: The category of the named entities.
            :return: Set of named entities.
            """

            return frozenset(
                entity.lemma_.split()[-1]
                for entity in text.ents
                if entity.label_ == label and len(entity.lemma_) > 0
            )

        # Extract the named entities from the candidate
        candidate_entities = extract_entities(candidate[1], self.label)

        ious = []
        for _, article in articles:
            # Extract the entities
            entities = extract_entities(article, self.label)

            # Calculate the intersection over union
            union = len(candidate_entities.union(entities))
            ious.append(
                len(candidate_entities.intersection(entities)) / union
                if union > 0
                else 0
            )

        return NamedEntities.aggregate(ious)
