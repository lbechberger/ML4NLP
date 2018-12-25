import pandas as pd
import numpy as np

def main():
	dgen = data_gen(n=19000)
	articles = next(dgen, [])
	n = 1
	while len(articles) > 0:
		print("--------> Iteration: " + str(n))
		enrich(articles, n)
		articles = next(dgen, [])
		n += 1
	print("Successfully processed all articles.")


def enrich(articles, iteration):
	n = 1
	for article_uri in articles["article_uri"].unique():
		print("Iteration {}, article {}.".format(iteration, n))
		article_cdata = []
		triple_indices = list(np.where(articles["article_uri"] == article_uri)[0])
		filename = article_uri.split("/")[-1]
		article_csv = pd.read_csv("processed_texts/{}.csv".format(filename), index_col=0)
		word_ref = article_csv.values
		for triple_index in triple_indices:
			char = articles.loc[triple_index]["word_start_char"]
			if char in article_csv.index:
				row = {}
				relation_info = article_csv.loc[char].values
				row["classification"] = articles.loc[triple_index]["classification"]
				row["agent_word"] = articles.loc[triple_index]["agent"]
				row["patient_word"] = articles.loc[triple_index]["patient"]
				row["relation_word"] = relation_info[2]
				row["relation_position"] = relation_info[1]
				ref_index = (word_ref == relation_info).all(axis=1).nonzero()[0][0]
				for diff in range(1, 11):
					plus_index = ref_index + diff
					if plus_index < len(word_ref) and word_ref[plus_index][0] < relation_info[0] + 2:
						row["+{}_word".format(diff)] = word_ref[plus_index][2]
						row["+{}_position".format(diff)] = word_ref[plus_index][1]
					else:
						row["+{}_word".format(diff)] = ""
						row["+{}_position".format(diff)] = -999
					minus_index = ref_index - diff
					if minus_index < len(word_ref) and word_ref[minus_index][0] > relation_info[0] - 2:
						row["-{}_word".format(diff)] = word_ref[minus_index][2]
						row["-{}_position".format(diff)] = word_ref[minus_index][1]
					else:
						row["-{}_word".format(diff)] = ""
						row["-{}_position".format(diff)] = -999
				article_cdata.append(row)
		df = pd.DataFrame(article_cdata)
		df.to_csv("articles_cdata/{}.csv".format(filename))
		n += 1


def data_gen(n=19000):
	for findex in range(0, n, 1000):
		yield pd.read_csv("merged_data/cdata_articles_{}_to_{}.csv.zip".format(findex, findex + 999), index_col=0)


if __name__ == "__main__":
	main()
