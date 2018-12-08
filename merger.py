import pandas as pd


def main():
	start=19000
	stop=19749
	step=50
	conc = concatenate_dfs(start, stop, step)
	print(conc.info())
	print(conc.head())
	print(conc.describe())
	conc.to_csv("merged_data/cdata_articles_%s_to_%s.csv" % (start,stop))
	print("File successfully saved.")



def concatenate_dfs(start, stop, step):
	paths = ["generated_data/cdata_articles_{}_to_{}.csv".format(findex, findex + step - 1) for findex in range(start, stop + 1, step)]
	dfs = [pd.read_csv(p,index_col=0) for p in paths]
	return pd.concat(dfs, ignore_index=True, sort=True)


if __name__ == "__main__":
	main()
