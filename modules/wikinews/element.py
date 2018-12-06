class Element:
    """ An element with an unique URL on WikiNews. """

    def __init__(self, url: str):
        assert url.startswith("/wiki/"), "Invalid relative WikiNews URL '{}'!".format(
            url
        )
        self.url = url

    def get_url(self) -> str:
        """ Returns the absolute URL of the element. """
        return Element.convert_url(self.url)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.url

    @staticmethod
    def convert_url(relative_url: str) -> str:
        """ Convert a relative WikiNews URL to an absolute one. """
        return "https://en.wikinews.org{}".format(relative_url)

