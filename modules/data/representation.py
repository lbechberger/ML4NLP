import random
from typing import Dict, Set, List, Iterator, Tuple


class Representation:
    class Parameter:
        def __init__(
            self,
            num_articles_per_interest: int,
            num_positive_samples: int,
            num_negative_samples: int,
        ):
            self.num_articles_per_interest = num_articles_per_interest
            self.num_positive_samples = num_positive_samples
            self.num_negative_samples = num_negative_samples

        def generate(
            self,
            interests: Dict[str, Set["Article"]],
            user_interests: Dict[str, Set["Article"]],
        ) -> "Representation":
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
                read_articles.extend(interesting_article[self.num_positive_samples:])
                positive_samples.extend(
                    interesting_article[:self.num_positive_samples]
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
        self, articles: List["Article"], positive_samples: "Article", negative_samples: "Article"
    ):
        self.articles = articles
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

    def __str__(self) -> str:
        return "Articles: {}\n -> Interesting: {}\n -> Not interesting: {}".format(
            self.articles, self.positive_samples, self.negative_samples
        )

    def rows(self): # -> Iterator[Tuple[List["Article"], "Article", bool]]:
        """ Returns all the different inputs to the machine learning algorithm. """

        for positive_sample in self.positive_samples:
            yield [self.articles, positive_sample, True]
        for negative_sample in self.negative_samples:
            yield [self.articles, negative_sample, False]

