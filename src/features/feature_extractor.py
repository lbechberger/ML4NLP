from pydoc import locate
from typing import Sequence, Dict, List, Tuple, Any, Union
from abc import ABC, abstractmethod
from statistics import mean, median, variance, stdev
import re

from spacy.tokens import Doc


class FeatureExtractor(ABC):
    """
    A abstract base class for a feature extractor.
    """

    def prepare(self, unique_articles: Sequence[Doc]) -> None:
        """
        Prepare the extractor with all the articles it have to expect. Does nothing by default and may be overwritten.
        :param unique_articles: A sequence of unique articles the extractor has to expect.
        """
        pass

    @classmethod
    def get_num_features(cls) -> int:
        """
        Return the number of features returned by this classier. May be overwritten.
        :return: Number of returned features.
        """

        return 6

    @abstractmethod
    def __call__(
        self, articles: List[Tuple[int, Any]], candidate: Tuple[int, Any]
    ) -> Sequence[float]:
        """
        Extract the features.
        :param articles: List of articles and their ids.
        :param candidate: Article and the id representing the candidate.
        :return: The features with the size returned by 'get_num_features'.
        """

        raise NotImplementedError()

    @classmethod
    def from_name(
        cls, name: str, arguments: Union[Sequence, Dict, None]
    ) -> "FeatureExtractor":
        """
        Create a feature extractor by its name. The class is searched in the src.features dictionary.
        :param name: The name of the classifier.
        :param arguments: Potential arguments to the classifier.
        :return: A instance of the classifier initialized with the arguments.
        """

        full_name = "src.features.{}.{}".format(
            FeatureExtractor.string_to_snake_case(name), name
        )
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

    @classmethod
    def aggregate(cls, individual_scores: Sequence[float]) -> Sequence[float]:
        """
        Aggregate a sequence of random lenght to a fixed-size vector by extracting key properties.
        :param individual_scores: The sequence of random lenght.
        :return: The fixed-size representation.
        """

        return (
            mean(individual_scores),
            median(individual_scores),
            min(individual_scores),
            max(individual_scores),
            variance(individual_scores),
            stdev(individual_scores),
        )

    @staticmethod
    def string_to_snake_case(input: str) -> str:
        """
        Format a string as snake_case, for example to find the class 'ExampleClass' in file 'example_class'
        Adopted from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        :param input: The input string.
        :return: The input string formatted as snake_case.
        """

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", input)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
