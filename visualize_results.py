#%%
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from src import PATH_ROOT


def show_boxplot(data: pd.DataFrame) -> None:
    """
    Show a boxplot with the complete range of performances of the classifiers.
    :param data: The evaluation results of the classifier.
    """

    boxplots = data.groupby("Classifier").boxplot(layout=(1, 6)).to_numpy()

    figure = boxplots[0].get_figure()
    figure.suptitle("Performances archived by different types of classifiers")
    plt.show()


def show_best(data: pd.DataFrame) -> None:
    """
    Plot the best results of each classifier.
    :param data: The evaluation results of the classifier.
    """

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


if __name__ == "__main__":
    # Load the results from the CSV file
    results = pd.read_csv(PATH_ROOT / "results.csv", sep="\t")

    # Rename the colums of interest
    results = results.rename(
        columns={"param_classifier": "Classifier", "mean_test_score": "F1 score"}
    )

    # Remove unnecessary details of classifier.
    results["Classifier"] = results["Classifier"].apply(
        lambda classifier: classifier[: classifier.find("(")]
    )

    # Visualize the result
    show_best(results)
    show_boxplot(results)
