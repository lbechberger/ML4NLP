class Element:
    def __init__(self, url):
        self.url = url
    
    def get_url(self):
        return Element.convert_url(self.url)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return '"{}"'.format(self.url)

    @staticmethod
    def convert_url(relative_url):
        return 'https://en.wikinews.org{}'.format(relative_url)