import warnings, os
import pandas as pd
from generation_functions import *
from feature_extraction import FeatureExtraction
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix

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
f = FeatureExtraction(articles, categories)
features = f.get_features(users_db)
print("\nNumber of feature arrays: {} = {} user * {} articles".format(len(features), len(users_db), len(articles)))


"""
Dimension Reduction
Choose filter method here by uncommenting corresponding string. Might take longer if no pickle file is found
"""
filter_method = 'filter'  # 'wrapper'
filtered = f.reduce_dimension(features, labels, 10, filter_method)
print("Contains NaN values:", np.any(np.isnan(filtered)))  # check for NaN values

# set up classifiers
classifiers = [
    ('kNN', KNeighborsClassifier(n_neighbors=8, weights='distance', p=1)),
    ('RF', RandomForestClassifier(n_estimators=128, criterion='entropy'))
]

# split dataset into test and training data via k-fold
# and train with model
kf = KFold(n_splits=10, shuffle=True)

# parameter grids for grid search optimisation
parameter_grid_knn = {'n_neighbors': np.arange(1, 21), 'weights': ['uniform', 'distance'], 'p': [1, 1.5, 2]}
parameter_grid_rf = {'n_estimators': [8, 16, 32, 64, 128], 'criterion': ['gini', 'entropy']}

print("\nDimension Reduction based on {} Methods. Optimisation via Grid Search".format(filter_method))
# train with previously defined classifiers for each k-fold split
for name, model in classifiers:
    print("Start training with", name)
    # outfile = open("output.txt", "a") # writes results of hyperparameter optimisation in file
    kappa_before = []
    precision_before = []
    recall_before = []
    '''
    # Uncomment if doing hyperparameter optimisation
    kappa_after = []
    precision_after = []
    recall_after = []
    '''
    confusion_matrix_result = []
    # split dataset n_splits times (see above)
    for train_index, test_index in kf.split(filtered):
        X_train = []; y_train = []
        X_test = []; y_test = []
        # for each split, get data
        for i in train_index:
            X_train.append(filtered[i])
            y_train.append(labels[i])
        for j in test_index:
            X_test.append(filtered[j])
            y_test.append(labels[j])

        # with data of split i, train the model and calculate the metrics
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        confusion_matrix_result.append([tn, fp, fn, tp])
        kappa_before.append(cohen_kappa_score(y_test, predictions))
        precision_before.append(precision_score(y_test, predictions))
        recall_before.append(recall_score(y_test, predictions))

        '''
        Hyperparameter Optimisation
        This takes really long for only one split. Saves parameters of first split externally in output file
        Uncomment if want to optimise again
        '''
        ## depending on model, use different parameter grid
        # parameter_grid = parameter_grid_knn if name == 'kNN' else parameter_grid_rf
        # grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, scoring=make_scorer(cohen_kappa_score))
        # grid_search.fit(X_train, y_train)
        # print('Best parameters:', grid_search.best_params_)
        # outfile.write(('Best parameters for {}: {}'.format(name, grid_search.best_params_)))
        # outfile.close()
        # predictions = grid_search.predict(X_test)
        # kappa_after.append(cohen_kappa_score(y_test, predictions))
        # precision_after.append(precision_score(y_test, predictions))
        # recall_after.append(recall_score(y_test, predictions))
    ## take the mean over all splits
    # print("Before Grid-Search\n", name, np.mean(kappa_before), np.mean(precision_before), np.mean(recall_before))
    # print("After Grid-Search:", name, np.mean(kappa_after), np.mean(precision_after), np.mean(recall_after))

    # used to check performances when inserting optimised parameters for each split
    print("Confusion Matrix for {}:\n  TN     FP    FN     TP".format(name))
    for m in confusion_matrix_result:
        print(m)

