import random
import pickle
import json

import pandas as pd

from . import User, Representation
from pathlib import Path

class DataSet:
    class Parameter:
        def __init__(self, user_params, num_user):
            self.user_params = user_params
            self.num_user = num_user
            
        def generate(self, interests):
            return DataSet(
                [self.user_params.generate(interests) for _ in range(self.num_user)],
                self
            )

        @staticmethod
        def from_json(path):
            with open(path, 'r') as json_file:
                data = json.load(json_file)

                user_params = data['user']
                representation_params = user_params['representation']

                return DataSet.Parameter(
                    user_params=User.Parameter(
                        representation_params=Representation.Parameter(
                            num_articles_per_interest=representation_params['articles_per_interest'], 
                            num_positive_samples=representation_params['positive_samples'], 
                            num_negative_samples=representation_params['negative_samples']
                        ), 
                        num_interests=user_params['interest'], 
                        num_representations=user_params['representations']
                    ),
                    num_user=data['users']
                )

    DATASET_NAME =  "Dataset_{}"

    def __init__(self, users, hyperparameters):
        self.users = users
        self.params = hyperparameters
        
    def __str__(self):
        return "Dataset with {} users.".format(len(self.users))
    
    def save(self, path, filename="dataset.pickle"):
        num_existing_datasets = len([True for f in path.iterdir() if f.is_dir()])

        current_storage = path / DataSet.DATASET_NAME.format(num_existing_datasets)
        current_storage.mkdir()

        with open(current_storage / filename, "wb") as file:
            pickle.dump(self, file)
        return current_storage

    @staticmethod
    def get_last_dataset(path):
        candidates = {int(f.name.split('_')[1]): f for f in path.iterdir() if f.is_dir()}
        return candidates[max(candidates.keys())]

    @staticmethod
    def load(path, version = None, filename="dataset.pickle"):
        dataset_path = DataSet.get_last_dataset(path) if version is None else path / DataSet.DATASET_NAME.format(version)
        with open(dataset_path / filename, "rb") as file:
            return pickle.load(file)
        
    def as_dataframe(self):
        column_names = [
            "interest_{}".format(i)
            for i in range( 
                self.params.user_params.num_interests
            )
        ] + [
            "article_{}".format(i)
            for i in range(
                self.params.user_params.representation_params.num_articles_per_interest * 
                self.params.user_params.num_interests
            )
        ] + ["candidate", "label"]
        
        return pd.DataFrame.from_records((
            [str(interest) for interest in interests] + 
            [article.url for article in articles] + 
            [candidate.url] + 
            [label]
            for user in self.users 
            for interests, articles, candidate, label in user.rows()
        ), columns=column_names)