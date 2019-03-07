import json
from typing import Tuple, Sequence, Optional, Any, Callable

import requests

from .element import Element
from ..knowledgestore import ks


class Article(Element):
    """ An article available in the KnowledgeStore read by a user. """

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        categories: Sequence[str] = (),
        text: Optional[str] = None,
    ):
        """
        Create an article.
        :param url: The relative URL to the URL on WikiNews.
        :param name: The name of the article. If not set, this and following data is loaded from WikiNews.
        :param categories: The categories a article belongs to.
        :param text: The text of the article.
        """

        super().__init__(url)
        self.__parsed_text = None

        if name is not None:
            # Data does already exist. Do not load from KnowledgeStore and WikiNews.
            self.name = name.strip()
            self.categories = categories
            self.text = text
        else:
            self.name, self.text, self.categories = Article._parse(url)

    def is_meaningfull(self) -> bool:
        """
        Check if the article is semantically meaningful for a news reader.
        :return: True, if the article is not WikiNews meta data.
        """

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

    def parsed_text(self, parser: Callable[[str], Any]) -> Any:
        """
        Parse and cache the underlying text.
        :param parser: The parser used to parse the text.
        :return: The parsed cache.
        """

        if self.__parsed_text is None:
            self.__parsed_text = parser(self.text)
        return self.__parsed_text

    @staticmethod
    def _parse(url: str) -> Tuple[str, str, Sequence[str]]:
        """
        Load the article data from the KnowledgeStore and WikiNews.
        :param url: Relative url to the article.
        :return: Title, text and categories.
        """

        # Create an URL for the WikiNews API
        json_url = "https://en.wikinews.org/w/api.php?action=query&titles={}&prop=categories&format=json".format(
            url[6:]
        )

        # Load the data from WikiNews
        response = json.loads(requests.get(json_url, timeout=5).content)
        pages = response["query"]["pages"]
        if len(pages) != 1:
            raise RuntimeError("Invalid number of pages.")

        # Parse the returned object
        page = next(iter(pages.values()))
        categories = page["categories"] if "categories" in page else []

        # Load the text
        text = ks.run_files_query("http://en.wikinews.org{}".format(url))

        return (
            page["title"],
            text,
            tuple(category["title"][9:] for category in categories),
        )
