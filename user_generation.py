import pickle
import numpy as np
import pandas as pd
from generation_functions import *

# always use same numbers (for performance comparison)
np.random.seed(0)

# load categories, articles and URIs from file
categories, uris, _ = pickle.load(open("user_articles.pickle", "rb"))
# get texts of all articles. Will be dumped in pickle file to avoid unnecessary computations when re-running code
# articles = get_articles_from_link(uris)
articles = pickle.load(open("articles.pickle", "rb"))


# create a dataframe with 100 users (rows) who get randomly assigned 0 or 1 for each category (cols)
users_db = pd.DataFrame(np.random.randint(2, size=(100, len(categories))), columns=categories)

"""
For each user/row(axis=1): correct subcategories' label to 0 if supercategory is disliked. To verify correctness
of function, use the following print function and check i.e. (sub)category "Transport":
Aviation before 1, after 0)
# print(users_db.loc[0], "\n")
"""
users_db.apply(lambda row: check_subcategories(row), axis=1)

input_data, labels = create_dataset(users_db, articles, 100)

# for d in input_data[98:103]:
#     print(d)
# for l in labels[98:103]:
#     print(l)

print("\nThere are {} categories. In total {} articles resulting in a dataset length of {}"
      .format(len(categories), len(articles), len(input_data)))
