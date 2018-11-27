import pickle
import numpy as np
import pandas as pd

categories, articles, _ = pickle.load(open("user_articles.pickle", "rb"))

np.random.seed()

print("\nThere are {} categories: {}\n".format(len(articles), categories))
aCount = 0
for c in articles:
    aCount += len(c)

print("Total number of articles: {}".format(aCount))
print([cat[0] for cat in articles]) # print first article of each category

# create a dataframe with 100 users (rows) who get randomly assigned 0 or 1 for each category (cols)
users_db = pd.DataFrame(np.random.randint(2, size=(100, len(categories))), columns=categories)
# print(users_db.loc[0]) # vertical print
# print(users_db.loc[[0]])


# wenn obercat loop durch untercat und allen eine  0  assignen -> alle untercat auch 0. Wenn 1 -> random likes so lassen
# user sind representiert als array of likes 