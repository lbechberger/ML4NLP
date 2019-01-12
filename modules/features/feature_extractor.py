from pydoc import locate
from typing import Sequence, Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import re

class FeatureExtractor(ABC):
    def prepare(self, unique_articles):
        pass

    @classmethod
    def get_num_features(cls):
        return 4

    @abstractmethod
    def __call__(self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]):
        raise NotImplementedError()

    @classmethod
    def from_name(cls, name: str, arguments):
        full_name = "modules.features.{}.{}".format(FeatureExtractor.string_to_snake_case(name), name)
        feature_class = locate(full_name)

        if feature_class is None:
            raise RuntimeError("Unable to load '{}'.".format(full_name))
        elif not issubclass(feature_class, cls):
            raise RuntimeError("The feature '{}' is not valid.".format(full_name))

        if isinstance(arguments, Sequence):
            return feature_class(*arguments)
        elif isinstance(arguments, Dict):
            return feature_class(**arguments)
        else:
            return feature_class()

    @staticmethod
    def string_to_snake_case(input: str) -> str:
        # Adopted from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', input)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
