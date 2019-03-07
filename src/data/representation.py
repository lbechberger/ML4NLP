import random
from typing import Dict, Set, Sequence, Iterator, Tuple


class Representation:
    """
    One of the representations of a user visible to the classifier.
    """

    class Parameter:
        """
        The parameters to create a representation.
        """

        def __init__(
            self,
            num_articles_per_interest: int,
            num_positive_samples: int,
            num_negative_samples: int,
        ):
            """
            Create a new object with the parameters for a representation.
            :param num_articles_per_interest: Number of articles per interest embedded into the representation.
            :param num_positive_samples: The number of positive samples.
            :param num_negative_samples: The number of negative samples.
            """

            self.num_articles_per_interest = num_articles_per_interest
            self.num_positive_samples = num_positive_samples
            self.num_negative_samples = num_negative_samples

        def generate(
            self,
            interests: Dict[str, Set["Article"]],
            user_interests: Dict[str, Set["Article"]],
        ) -> "Representation":
            """
            Create a representation from available interest and parameter.
            :param interests: The available interest a user may have.
            :param user_interests: The interest of a user represented by the resulting representation.
            :return: A representation of a user.
            """

            interesting_category = [
                random.sample(
                    interests[interest],
                    self.num_articles_per_interest + self.num_positive_samples,
                )
                for interest in user_interests
            ]

            read_articles = []
            positive_samples = []
            negative_samples = []

            # Generate positive samples
            for interesting_article in interesting_category:
                read_articles.extend(interesting_article[self.num_positive_samples :])
                positive_samples.extend(
                    interesting_article[: self.num_positive_samples]
                )

            # Generate negative samples
            while len(negative_samples) < self.num_negative_samples:
                negative_category = random.choice(tuple(interests.keys()))
                if negative_category not in user_interests:
                    candidates = set(interests[negative_category])
                    for user_interest in user_interests:
                        candidates = candidates.difference(interests[user_interest])

                    if (
                        len(candidates) + len(negative_samples)
                        <= self.num_positive_samples
                    ):
                        negative_samples.extend(candidates)
                    elif len(candidates) > 0:
                        limit = self.num_negative_samples - len(negative_samples)
                        negative_samples.extend(
                            (c for i, c in enumerate(candidates) if i < limit)
                        )

            # Sanity checks
            assert len(positive_samples) == self.num_positive_samples * len(
                user_interests
            ), "Invalid number of positiv samples"
            assert (
                len(negative_samples) == self.num_negative_samples
            ), "Invalid number of negativ samples"

            return Representation(
                articles=read_articles,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
            )

    def __init__(
        self,
        articles: Sequence["Article"],
        positive_samples: Sequence["Article"],
        negative_samples: Sequence["Article"],
    ):
        """
        Create a representation. You should use "Representation.Parameter.generate"!
        :param articles: The articles embedded in the representation.
        :param positive_samples: The article matching the underlying interest.
        :param negative_samples: The article not matching the underlying interest.
        """

        self.articles = articles
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

    def __str__(self) -> str:
        return "Articles: {}\n -> Interesting: {}\n -> Not interesting: {}".format(
            self.articles, self.positive_samples, self.negative_samples
        )

    def samples(self) -> Iterator[Tuple[Sequence["Article"], "Article", bool]]:
        """
        Returns all the different samples to the machine learning algorithm.
        :return: Iterator generating positive and negative samples for a specific representation.
        """

        for positive_sample in self.positive_samples:
            yield (self.articles, positive_sample, True)
        for negative_sample in self.negative_samples:
            yield (self.articles, negative_sample, False)
