import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, accuracy_score


def main():
	print("Reading data ...")
	data = read_data()
	print("Data successfully read.")
	y = data.classification.copy()
	X = data[["word_end_char", "word_start_char"]]
	splitter = StratifiedShuffleSplit(n_splits=10, random_state=42)
	splits = splitter.split(X, y)
	always_false = DummyClassifier(strategy="constant", constant = 0)
	precision = score(always_false, next(splits), X, y)
	print(precision)



def score(classifier, split, X, y):
	train_index, test_index = split
	train_X = X.iloc[train_index]
	train_y = y.iloc[train_index]
	test_X = X.iloc[test_index]
	test_y = y.iloc[test_index].apply(int).values
	classifier.fit(train_X, train_y)
	predictions = classifier.predict(test_X)
	return accuracy_score(y_true=test_y, y_pred=predictions)



def read_data(n=19000):
	paths = ["merged_data/cdata_articles_{}_to_{}.csv.zip".format(findex, findex + 999) for findex in range(0, n, 1000)]
	return pd.concat([pd.read_csv(p, index_col=0) for p in paths], ignore_index=True, sort=True)


if __name__ == "__main__":
	main()
