import knowledgestore.ks as ks
import pandas as pd
import numpy as np
import time, pickle, csv

top_level_categories = ["Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business", 
                        "Environment", "Health", "Science and technology", "Sports", "Wackynews", "Weather", 
                        "Politics and conflicts", "Obituaries", "Transport", "World", "Internet", "Religion" ]

# if first run: uncomment following lines and comment 15+16
#listOfArticles = []
# with open('all_article_uris.csv', 'r') as csvFile:  
#     allArticles = list(csv.reader(csvFile)) # read in all articles as list

with open("user_articles.pickle",'rb') as ppf: 
        _, listOfArticles, allArticles = pickle.load(ppf)


# listOfArticles : nested list of categories' articles, 
# e.g. listOfArticles[0][0] = first article in cat Crime and law
# save set of possible articles in allArticles (because need to "remember" dumped ones)

for c in top_level_categories[len(listOfArticles):]:
	# save articles for each category in a separate array which will be appended to the listOfArticles array later on
	catArticles = []
	articleCounter = 0
	while (articleCounter < 100):
		randomRow = np.random.randint(1, len(allArticles))
		newArticle = allArticles[randomRow][0] 
		# if article belongs to c
		if (ks.get_applicable_news_categories(newArticle, [c]) != []): 
			catArticles.append(newArticle)
			allArticles.pop(randomRow) # remove from list of possible articles to avoid duplicates
			articleCounter += 1
		# if article does not belong in this category try again
		else:
			continue

	listOfArticles.append(catArticles)
	print("Finished {}".format(c))

	# save data to pickle
	with open('user_articles.pickle', 'wb') as pf:
		pickle.dump([top_level_categories[0:len(listOfArticles)], listOfArticles, allArticles], pf)

