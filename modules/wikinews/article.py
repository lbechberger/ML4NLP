import json
from typing import Tuple, List, Union

import requests

from .element import Element


class Article(Element):
    """ An article available in the KnowledgeStore read by a user. """

    def __init__(
        self, url: str, name: Union[str, None] = None, categories: List[str] = []
    ):
        super().__init__(url)

        if name is not None:
            self.name = name.strip()
            self.categories = categories
        else:
            self.name, self.categories = Article.parse(url)

    def is_meaningfull(self) -> bool:
        """ Check if the article is semantically meaningfull for a news reader. """

        if not self.url.startswith("/wiki/"):
            return False
        elif (
            self.name.startswith("Template")
            or self.name.startswith("Portal")
            or self.name.startswith("User")
            or self.name.startswith("Talk")
            or self.name.startswith("Help")
            or self.name.startswith("TEST")
            or self.name.startswith("Category")
            or self.name.startswith("Module")
            or self.name.startswith("News")
        ):
            return False
        elif "Wikinews" in self.name:
            return False
        else:
            return True

    @staticmethod
    def parse(url: str) -> Tuple[str, List[str]]:
        """ Generates an article from a given URL. """

        # Create an URL for the WikiNews API
        json_url = "https://en.wikinews.org/w/api.php?action=query&titles={}&prop=categories&format=json".format(
            url[6:]
        )

        # Load the data from WikiNews
        response = json.loads(requests.get(json_url, timeout=5).content)
        pages = response["query"]["pages"]
        if len(pages) != 1:
            raise "Invalid number of pages."

        # Parse the returned object
        page = next(iter(pages.values()))
        categories = page["categories"] if "categories" in page else []
        return page["title"], [category["title"][9:] for category in categories]
