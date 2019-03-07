import random
from typing import Dict, Set, Sequence, Iterator, Tuple


class User:
    """
    The underlying user with specific interests.
    """

    class Parameter:
        """
        The parameters to create a user.
        """

        def __init__(
            self,
            representation_params: "Representation.Parameter",
            num_interests: int,
            num_representations: int,
        ):
            """
            Create a new object with the parameters for a user.
            :param representation_params: The parameters for the underlying representations.
            :param num_interests: The number of interests of the user.
            :param num_representations: The number of underlying representations.
            """

            self.num_interests = num_interests
            self.num_representations = num_representations
            self.representation_params = representation_params

        def generate(self, interests: Dict[str, Set["Article"]]) -> "User":
            """
            Create a user from available interests.
            :param interests: The available interests.
            :return: A new user.
            """

            user_interests = random.sample(interests.keys(), self.num_interests)
            representations = [
                self.representation_params.generate(interests, user_interests)
                for _ in range(self.num_representations)
            ]
            return User(interests=user_interests, representations=representations)

    def __init__(
        self, interests: Sequence[str], representations: Sequence["Representation"]
    ):
        """
        Create a new user. You should use "User.Parameter.generate"!
        :param interests: The interests of a user.
        :param representations: The underlying representations.
        """

        self.interests = interests
        self.representations = representations

    def __str__(self) -> str:
        return "User with {} interests ({} representations): {}".format(
            len(self.interests), len(self.representations), self.interests
        )

    def samples(
        self
    ) -> Iterator[Tuple[Sequence[str], Sequence["Article"], "Article", bool]]:
        """
        Returns all the different samples to the machine learning algorithm.
        :return: Iterator generating positive and negative samples for a specific user.
        """

        for representation in self.representations:
            for row in representation.samples():
                yield (self.interests, *row)
