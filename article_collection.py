import knowledgestore.ks as ks
import numpy as np
import pickle, csv
import os.path

top_level_categories = ["Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business",
                        "Environment", "Health", "Science and technology", "Sports", "Wackynews", "Weather",
                        "Politics and conflicts", "Obituaries", "Transport", "World", "Internet", "Religion",
                        "Africa", "Asia", "Europe", "Middle East", "North America", "Oceania", "Aviation", "Computing",
                        "Space", "Elections", "Human rights", "United Nations", "China", "India", "Russia", "Iraq",
                        "Israel", "Australia", "New Zealand", "California", "United Kingdom", "France", "England",
                        "London"]

# if there already exists a pickle file: append it, otherwise create new one
if os.path.isfile('./user_articles.pickle'):
    with open("user_articles.pickle", 'rb') as ppf:
        _, listOfArticles, allArticles = pickle.load(ppf)
else:
    with open('all_article_uris.csv', 'r') as csvFile:
        # read in all articles as list
        allArticles = list(csv.reader(csvFile))
        listOfArticles = []

# listOfArticles = nested list of categories' articles
# e.g. listOfArticles[0][0] = first article in cat Crime and law
# save set of possible articles in allArticles (because need to "remember" dumped ones)

for c in top_level_categories[len(listOfArticles):]:
    # save articles for each category in a separate array which will be appended to the listOfArticles array later on
    catArticles = []
    articleCounter = 0
    while articleCounter < 100:
        randomRow = np.random.randint(1, len(allArticles))
        newArticle = allArticles[randomRow][0] 
        # if article belongs to c
        if ks.get_applicable_news_categories(newArticle, [c]) != []:
            catArticles.append(newArticle)
            # remove from list of possible articles to avoid duplicates
            allArticles.pop(randomRow)
            articleCounter += 1
        else:
            # if article does not belong in this category try again
            continue

    listOfArticles.append(catArticles)
    print("Finished {}".format(c))

    # save data to pickle
    with open('user_articles.pickle', 'wb') as pf:
        pickle.dump([top_level_categories[0:len(listOfArticles)], listOfArticles, allArticles], pf)

