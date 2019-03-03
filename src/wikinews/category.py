import re
from typing import Union, Dict, List, Tuple, Set

import requests
from bs4 import BeautifulSoup

from .element import Element
from .article import Article


class Category(Element):
    """ A category with probably multiple attached articles. """

    def __init__(
        self,
        url: str,
        name: Union[str, None] = None,
        sub_categories: List[str] = [],
        articles: List[str] = [],
    ):
        super().__init__(url)
        self.name = name
        self.sub_categories = sub_categories
        self.articles = articles

    def parse(
        self,
        articles_cache: Dict[str, Article],
        categories_cache: Dict[str, "Category"],
        check_article: bool = True,
        check_categories: bool = True,
    ) -> bool:
        """ 
        This function enriches the category with the corresponding articles and subcategories. 
        TODO: Move to JSON API
        """

        page_html = requests.get(self.get_url(), timeout=5)
        page_content = BeautifulSoup(page_html.content, "html.parser")

        if self.name is None:
            title_element = page_content.find("h1")
            if title_element is not None:
                self.name = title_element.text[9:].strip()

        # Parse and add unknown articles
        mw_pages = page_content.find("div", {"id": "mw-pages"})
        if mw_pages is not None:
            article_candidates = [
                Article(name=article.text, url=article["href"])
                for article in mw_pages.findAll("a")
            ]
            articles = {
                article.url: article
                for article in article_candidates
                if (not check_article) or article.is_meaningfull()
            }
            articles_cache.update(articles)

            # Add childs
            self.articles = list(articles.keys())

        # Parse and add subcategories
        mw_subcategories = page_content.find("div", {"id": "mw-subcategories"})
        if mw_subcategories is not None:
            self.sub_categories = [
                category["href"]
                for category in mw_subcategories.findAll("a")
                if (not check_categories) or category["href"] in categories_cache
            ]

            if not check_categories:
                new_categories = [
                    Category(category_url)
                    for category_url in self.sub_categories
                    if category_url not in categories_cache
                ]

                categories_cache.update(
                    {
                        category.url: category
                        for category in new_categories
                        if category.parse(
                            articles_cache,
                            categories_cache,
                            check_article,
                            check_categories,
                        )
                    }
                )

        return True

    def __str__(self) -> str:
        return '"{}":\n\tArticles: {}\n\tSubcategories: {}\n'.format(
            self.name, self.articles, self.sub_categories
        )

    @staticmethod
    def from_urls(
        categories: List[str]
    ) -> Tuple[List["Category"], Dict[str, "Category"], Dict[str, Article]]:
        """ Load categories from a list of existing URLs. """

        articles_tmp = {}
        categories_tmp = {}
        return (
            [
                category
                for category in (Category(category) for category in categories)
                if category.parse(articles_tmp, categories_tmp, False, False)
            ],
            categories_tmp,
            articles_tmp,
        )

    @staticmethod
    def from_wikinews(
        limit: int = 2000, num_categories: int = -1
    ) -> Tuple[List["Category"], Dict[str, "Category"], Dict[str, Article]]:
        """ Load categories from all WikiNews categories. """

        WIKINEWS = "http://en.wikinews.org/w/index.php?title=Special:Categories&limit={}&offset={}"

        category_detector = re.compile(r"\/wiki\/Category:.+")
        number_detector = re.compile(r"\(([0-9,]+) ")
        categories = []

        # Iterate through pages
        while num_categories == -1 or len(categories) < num_categories:
            url = WIKINEWS.format(
                limit, categories[-1].url[15:] if len(categories) > 0 else ""
            )
            page_response = requests.get(url, timeout=5)
            page_content = BeautifulSoup(page_response.content, "html.parser").find(
                "div", {"class": "mw-spcontent"}
            )

            # Find all subcategories
            for candidate in page_content.find_all("a", {"href": category_detector}):
                num_childs = int(
                    number_detector.search(candidate.parent.text)
                    .group(1)
                    .replace(",", "")
                )
                if num_childs > 0:
                    categories.append(candidate["href"])

            if page_content.find("a", {"class": "mw-nextlink"}) is None:
                break

        return Category.from_urls(categories)

    @staticmethod
    def filter_categories(
        categories: Dict[str, Set[Article]], min_articles: int
    ) -> Dict[str, Set[Article]]:
        """ Filter categories for meaningfulness and a minimal amount of articles. """

        date_regex = re.compile(
            "^((January)|(February)|(March)|(April)|(May)|(June)|(July)|(August)|(September)|(October)|(November)|(December))"
        )
        year_regex = re.compile("^[0-9]{4}$")
        wiki_regex = re.compile(".*[w|W]iki.*")
        wiki_other_noise = [
            "archived",
            "dialog ",
            "files",
            "sockpuppets",
            "sources",
            "requests for",
            "peer reviewed",
            "non-",
            "articles",
            "media",
            "categor",
            "cc",
            "assistant",
            "pages",
            "checkuser",
            "template",
            "wwc",
            "failed",
            "abandoned",
            "local",
            "live",
            "wikinewsie",
            "user",
            "uow",
            "no publish",
            "published",
            "autoarchived",
            "original reporting",
            "audio reports",
            "out of date stories",
            "news of the world",
            "writing contest",
            "source offline",
            "photo essays",
            "football match reports",
        ]

        filtered = {
            key: value
            for key, value in categories.items()
            if not date_regex.match(key)
            and not year_regex.match(key)
            and not wiki_regex.match(key)
            and not any(True for phase in wiki_other_noise if phase in key.lower())
        }

        return {
            key: value for key, value in filtered.items() if len(value) >= min_articles
        }
