import pandas as pd


def main():
	data = read_data()
	print(data.info())
	print(data.head())
	print(data.describe())
	print(data["classification"].value_counts())


def read_data(n=5000):
	paths = ["merged_data/cdata_articles_{}_to_{}.csv.zip".format(findex, findex + 999) for findex in range(0, n, 1000)]
	return pd.concat([pd.read_csv(p, index_col=0) for p in paths], ignore_index=True, sort=True)


if __name__ == "__main__":
	main()
