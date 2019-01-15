from modules.features import FeatureExtractor


class NamedEntities(FeatureExtractor):
    def __init__(self, label):
        self.label = label

    def __call__(self, articles, candidate):
        def extract_entities(text, label):
            return frozenset(
                entity.lemma_.split()[-1]
                for entity in text.ents
                if entity.label_ == label and len(entity.lemma_) > 0
            )

        candidate_entities = extract_entities(candidate[1], self.label)
        ious = []
        for _, article in articles:
            entities = extract_entities(article, self.label)
            union = len(candidate_entities.union(entities))
            ious.append(
                len(candidate_entities.intersection(entities)) / union if union > 0 else 0
            )

        return NamedEntities.aggregate(ious)
