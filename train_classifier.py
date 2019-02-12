import sys
import itertools
import copy
from collections import OrderedDict
from typing import Sequence

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

sys.path.append("..")
from src import PATH_DATASETS, PATH_CONFIG
from src.load_features import Features



def _generate_transformer_versions(transformer, **kwargs):
    """
    This function generates all possible permutations of a tranformer with mutliple arguments.
    :param transformer: The class of the transformer
    :param kwargs: The arguments of the transformer. Sequence are interpreted as different versions.
    :return: Generator of Transformer instances
    """

    fixed_arguments = dict(
        (name, argument)
        for name, argument in kwargs.items()
        if not isinstance(argument, Sequence)
    )

    free_arguments = OrderedDict(
        (name, argument)
        for name, argument in kwargs.items()
        if isinstance(argument, Sequence)
    )

    for combination in itertools.product(*free_arguments.values()):
        arguments = copy.deepcopy(fixed_arguments)
        for argument_name, value in zip(free_arguments.keys(), combination):
            arguments[argument_name] = value
        yield transformer(**arguments)


if __name__ == "__main__":
    NUM_FEATURES = [5, 10, 15, 20]

    pipeline = Pipeline(
        [("preprocessing", None), ("reduce_dim", None), ("classifier", None)]
    )

    param_grid = {
        "preprocessing": [None, StandardScaler()],
        "reduce_dim": list(
            itertools.chain(
                [None],
                _generate_transformer_versions(
                    PCA, iterated_power=7, n_components=NUM_FEATURES
                ),
            )
        ),
        "classifier": list(
            itertools.chain(
                _generate_transformer_versions(
                    RandomForestClassifier, n_estimators=[8, 16, 32, 64, 128]
                ),
                _generate_transformer_versions(
                    ExtraTreesClassifier, n_estimators=[8, 16, 32, 64, 128]
                ),
                [
                    GaussianNB(),
                    QuadraticDiscriminantAnalysis(),
                    KNeighborsClassifier(3),
                    SVC(gamma="auto"),
                ],
            )
        ),
    }

    # Generate or load the features
    features = Features("features")
    X, y, split, X_test, y_test = features.load(dataset_path=PATH_DATASETS, feature_config_path=PATH_CONFIG / "features.json")

    # Do the grid search
    grid = GridSearchCV(
        pipeline,
        cv=PredefinedSplit(split),
        n_jobs=-1,
        param_grid=param_grid,
        scoring="f1",
        error_score="raise",
        return_train_score=True,
    )
    grid.fit(X, y)

    # Output the results
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values("rank_test_score", inplace=True, ascending=True)
    results.to_csv("results.csv", "\t", index=False)

    print("Results on test set:")
    print(confusion_matrix(y_test, grid.predict(X_test)))