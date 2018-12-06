import pandas as pd


def main():
	# dfs = []
	# dfs.append(pd.read_csv("merged_data/cdata_articles_1000_to_1299.csv", index_col=0))
	# dfs.append(pd.read_csv("merged_data/cdata_articles_1300_to_1899.csv", index_col=0))
	# dfs.append(pd.read_csv("merged_data/cdata_articles_1900_to_1999.csv", index_col=0))
	# conc = pd.concat(dfs, ignore_index=True, sort=True)
	conc = concatenate_dfs(2600, 2999, 50)
	print(conc.info())
	print(conc.head())
	print(conc.describe())
	print(len(conc.index))
	# print(sum([len(p.index) for p in dfs]))
	conc.to_csv("merged_data/cdata_articles_2600_to_2999.csv")
	print("File successfully saved.")


def concatenate_dfs(start, stop, step):
	paths = ["generated_data/cdata_articles_{}_to_{}.csv".format(findex, findex + step - 1) for findex in range(start, stop + 1, step)]
	dfs = [pd.read_csv(p, index_col=0) for p in paths]
	return pd.concat(dfs, ignore_index=True, sort=True)


if __name__ == "__main__":
	main()
