""" Uses the scripts contained within explorer.py to generate training and test data."""

import pandas as pd
import numpy as np
from knowledgestore import ks
import explorer
import copy
from logging import log

all_article_uris = explorer.all_article_uris
QUESTION_SIGN = "?"


def generate_all_data():
    all_data = pd.DataFrame()
    with open("data.csv", "w+") as file:
        for uri in all_article_uris["article"]:  # Todo: Parralize
            next_data = generate_data_from_uri(uri)
            if not len(next_data) == 0:
                all_data = all_data.append(next_data, ignore_index=True)

                all_data.to_csv(file)


def generate_data_from_uri(uri):
    answers = explorer.get_triplets(uri)
    if len(answers) == 0:
        return []
    questions = np.array(answers)
    questions[:, 1] = QUESTION_SIGN
    result = pd.DataFrame()
    for QAPair in range(len(answers)):
        result = result.append(
            {"answer": answers[QAPair], "question": questions[QAPair], "text": ks.run_files_query(uri), "uri": uri},
            ignore_index=True)
    return result


if __name__ == "__main__":
    generate_all_data()
