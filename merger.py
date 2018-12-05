import pandas as pd


def main():
	conc = concatenate_dfs(800, 999, 10)
	print(conc.info())
	print(conc.head())
	print(conc.describe())
	conc.to_csv("merged_data/cdata_articles_800_to_999.csv")
	print("File successfully saved.")


def concatenate_dfs(start, stop, step):
	paths = ["generated_data/cdata_articles_{}_to_{}.csv".format(findex, findex + step - 1) for findex in range(start, stop + 1, step)]
	dfs = [pd.read_csv(p,index_col=0) for p in paths]
	return pd.concat(dfs, ignore_index=True, sort=True)


if __name__ == "__main__":
	main()
