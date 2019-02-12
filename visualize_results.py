#%%
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from src import PATH_ROOT


def show_boxplot(data: pd.DataFrame):
    boxplots = data.groupby("Classifier").boxplot(layout=(1, 6)).to_numpy()

    figure = boxplots[0].get_figure()
    figure.suptitle("Performances archived by different types of classifiers")
    plt.show()


def show_best(data: pd.DataFrame):
    best_results = list(data.groupby("Classifier").max().itertuples())
    best_results.sort(key=lambda x: x[1])

    plt.barh(
        range(len(best_results)),
        [result[1] for result in best_results],
        tick_label=[result[0] for result in best_results],
    )
    plt.xlim(0.6, 0.85)
    plt.title("Best performance by classifier on validation set")
    plt.xlabel("F1-score")
    plt.show()

#%%

results = pd.read_csv(PATH_ROOT / "results.csv", sep="\t")
results = results[["param_classifier", "mean_test_score"]].rename(
    columns={"param_classifier": "Classifier", "mean_test_score": "F1 score"}
)
results["Classifier"] = results["Classifier"].apply(
    lambda classifier: classifier[: classifier.find("(")]
)

show_best(results)
show_boxplot(results)
