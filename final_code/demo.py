from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
import numpy as np
import pandas as pd
import pickle

classifiers = ["knn.pkl", "max_ent.pkl", "decision_tree.pkl", "random_forest.pkl"]

def evaluate(model, test_features, test_labels):
    """
    Copied this little helper from towardsdatascience.com
    """
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


if __name__ == '__main__':
    dataset = pd.read_csv("dataset_final(reduced_features).csv", sep=';')

    y = dataset.label

    data_vec = dataset.drop(['label'], axis = 1)
    data_vec.fillna('NA', inplace = True)
    data_vec = data_vec.to_dict(orient = 'records')
    vectorizer = DV(sparse = True)
    vec_data = vectorizer.fit_transform(data_vec)

    x = vec_data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42, shuffle = True)

    for classifier in classifiers:
        classifier_loc = r"classifiers/" + str(classifier)
        with open(classifier_loc, 'rb') as f:
            model = pickle.load(f)
        evaluate(model, x_test, y_test)


def print_help():
    print("When running this script, you can chose not to specify which classifier to use. In that case, it will use all four.")