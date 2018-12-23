import pandas as pd
from knowledgestore import ks
import time


def main():
	dgen = data_gen()
	data = next(dgen)
	n = 1
	while len(data) > 0:
		print("-------> Iteration " + str(n))
		article_uris = data["article_uri"].unique()
		download(article_uris)
		data = next(dgen, [])
		n += 1
	print("Successfully downloaded all articles.")


def download(article_uris):
	start = time.time()
	for article_uri in article_uris:
		article_text = ks.run_files_query(article_uri)
		filename = article_uri.split("/")[-1] + ".txt"
		with open("downloaded_texts/" + filename, "w+") as fh:
			fh.write(article_text)
	print("{0:0.1f} seconds elapsed and ".format(time.time() - start) + str(len(article_uris)) + " articles were downloaded.")


def data_gen(n=19000):
	for findex in range(0, n, 1000):
		yield pd.read_csv("merged_data/cdata_articles_{}_to_{}.csv.zip".format(findex, findex + 999), index_col=0)


if __name__ == "__main__":
	main()
