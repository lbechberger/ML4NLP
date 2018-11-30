import random

class User:
    class Parameter:
        def __init__(self, representation_params, num_interests, num_representations):
            self.num_interests  = num_interests
            self.num_representations = num_representations
            self.representation_params = representation_params
            
        def generate(self, interests):
            user_interests = random.sample(interests.keys(), self.num_interests)
            representations = [
                self.representation_params.generate(interests, user_interests)
                for _ in range(self.num_representations)
            ]
            return User(interests=user_interests, representations=representations)
        
    def __init__(self, interests, representations):
        self.interests = interests
        self.representations = representations
        
    def __str__(self):
        return "User with {} interests ({} representations): {}".format(
            len(self.interests),
            len(self.representations),
            self.interests
        )
    
    def rows(self):
        for representation in self.representations:
            for row in representation.rows():
                yield [self.interests] + row