import warnings, os
import pandas as pd
from generation_functions import *
from feature_extraction import FeatureExtraction
from sklearn.model_selection import KFold

# suppress warnings (usually deprecated warnings from classifier) & always use same numbers (for performance comparison)
warnings.filterwarnings("ignore")
np.random.seed(0)

# load categories, articles and URIs from file
categories, uris, _ = pickle.load(open("./data/user_articles.pickle", "rb"))
# get texts of all articles. Will be dumped in pickle file to avoid unnecessary computations when re-running code
if os.path.isfile('./data/articles.pickle'):
    articles = pickle.load(open("./data/articles.pickle", "rb"))
else:
    articles = get_articles_from_link(uris)

# create a dataframe with 100 users (rows) who get randomly assigned 0 or 1 for each category (cols)
# correct subcategory labels if superior category is disliked (=0)
users_db = pd.DataFrame(np.random.randint(2, size=(100, len(categories))), columns=categories)
users_db.apply(lambda row: check_subcategories(row), axis=1)

_, labels = create_dataset(users_db, articles, 100)

# Extract the features
f = FeatureExtraction(articles, categories, labels)
features = f.get_features(users_db)
print("\nNumber of feature arrays: {} user * {} articles = {}".format(len(users_db), len(articles), len(features)))


"""
Dimension Reduction
Choose filter method here by uncommenting corresponding string. Might take longer if no pickle file is found
"""
filter_method = 'filter'  # 'wrapper'
print("\nDimension Reduction based on {} Methods. Optimisation via Grid Search".format(filter_method))

filtered = f.reduce_dimension(features, labels, 10, filter_method)
print("Contains NaN values:", np.any(np.isnan(filtered)))  # check for NaN values

# split dataset into test and training data via k-fold
# and train with model
kf = KFold(n_splits=10, shuffle=True)

# training of classifier
f.train_classifier(filtered, kf, True, 2)

