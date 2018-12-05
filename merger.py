import pandas as pd

dfs = []
dfs.append(pd.read_csv("generated_data/merged/cdata_articles_0_to_741.csv", index_col=0))
dfs.append(pd.read_csv("generated_data/classification_data_chunks/cdata_articles_742_to_750.csv", index_col=0))
print(dfs[0].head())
print(dfs[1].head())
conc = pd.concat(dfs, ignore_index=True, sort=True)
print(conc.info())
print(conc.head())
print(conc.describe())
print(sum([len(d.index) for d in dfs]))
print(len(conc.index))
conc.to_csv("merged_data/cdata_articles_0_to_750.csv")
