import json

import requests

from .element import Element

class Article(Element):
    def __init__(self, url, name=None, categories=[]):
        super().__init__(url)
        
        if name is not None:
            self.name = name.strip()
            self.categories = categories
        else:
            self.name, self.categories = Article.parse(url)
    
    def is_meaningfull(self):
        if not self.url.startswith('/wiki/'):
            return False

        if self.name.startswith('Template') \
        or self.name.startswith('Portal') \
        or self.name.startswith('User') \
        or self.name.startswith('Talk') \
        or self.name.startswith('Help') \
        or self.name.startswith('TEST') \
        or self.name.startswith('Category') \
        or self.name.startswith('Module') \
        or self.name.startswith('News'):
            return False

        if 'Wikinews' in self.name:
            return False

        return True

    @staticmethod
    def parse(url):    
        json_url = "https://en.wikinews.org/w/api.php?action=query&titles={}&prop=categories&format=json".format(
            url[6:]
        )
        
        response = json.loads(requests.get(json_url, timeout=5).content)
        pages = response['query']['pages']
        if len(pages) != 1:
            raise ""
        page = next(iter(pages.values()))
        
        categories = page['categories'] if 'categories' in page else []
        return page['title'], [category['title'][9:] for category in categories]
            
    def __str__(self):
        return '"{}"'.format(self.name)