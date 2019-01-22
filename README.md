# ML4NLP - Group Beta
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Beta.

## Documentation Part 1: 

### Introduction
Today’s world changes at a rapid rate and we are flooded by news constantly. News apps update the user’s feed every minute and keeping up-to-date by oneself is a time-consuming challenge. To reduce the amount of information to the news the user is interested in, a News Recommendation System is required. Such a system is aimed to filter the articles to display solely the relevant reports. The News Recommendation System that will be implemented here, is going to filter news articles from the website wikinews.org. 

### Approaches
Building a News Recommendation System, there are two common approaches: collaborative and content-based approaches [1]. Collaborative or popularity-based approaches use the preferences of many users and similarity measures to predict whether or not a user will like a news article. Problems with this method arise mainly due to a lack of adequate data of user-preferences. Furthermore, it is hard to make predictions for new users, because of the lack of information concerning the user’s interests. For this approach it is therefore very helpful to have user feedback or ratings on articles to collect information about collective preferences, which wikinews.org does not provide. 

Content-based approaches take into account the similarity of the content of the news article and the content of the news articles the user previously liked. The content can be either represented by keywords or the categories an article is assigned to. In wikinews.org, every article belongs to one or more categories (which often also seem to be keywords).

A hybrid approach of a collaborative and a content-based approach could use existing user profiles that are based on collective user action. The user profile built by interpreting explicitly (questionnaire) or implicitly (tracing user activities) collected data can then be compared with the different user profiles from the collaborative approach and the user preferences of the (collective) profile closest to the user’s profile can additionally be taken into account to achieve better performance. If we can manage to obtain such collective user profiles on news, we will try to include it.

Because of the problems that come with the collaborative approach, we decided to implement a content-based news recommender system or a hybrid approach.

### Structure of our Approach [cf.1]: 
#### 1) Categories: 

Wikinews.org provides the following categories: 

Region: Africa • Asia • South / Central / North America • Europe • Middle East • Oceania • Antarctica

Topic: Crime and law • Culture and entertainment • Disasters and accidents • Economy and business • Education • Environment • Health • Obituaries • Politics and conflicts • Science and technology • Sports • Wackynews • Weather

We still need to decide on the scope of our recommender system and whether or not we want to include all categories, only the main categories, or only the subcategories of one main category, e.g. sports or economy

#### 2) Profile Building:

We decided to collect data explicitly by collecting user feedback of every new user. There are different possibilities to collect the data. One way is to show the user several articles from different categories, which are either randomly selected or explicitly chosen to be representative for a certain category, and let the user vote on them without showing the associated categories. Another possibility is to let the user directly vote on the categories in question. 

It may be possible to let the user choose which regional scope he/she is interested in regarding the different topics, e.g. Sports - worldwide, Crime and Law - Europe and Middle-East, Economy - Asia, Europe and North America, Weather - Berlin, etc. 
Also, when the user expresses an interest in Sports, it may be useful to further ask in what kind of sports (wikinews.org provides 59 subcategories), whereas for other categories the subcategories are less explicitly stated and should therefore not be queried.

#### 3) News Recommendation:

In the final step, we will give the recommender the user’s profile and the categorized articles as an input and receive recommendations based on their similarity as an output.


## Documentation Part 2 - Dataset: 

### News Articles and Categories

We are going to use the pre-existing dataset of news articles from Wikinews (https://www.wikinews.org). Every article possesses crowd-sourced labels of associated categories. The dataset is constantly manually updated with new articles and category labels. The dataset contains more than 21.000 articles in English, annotated with 22 main- and/or numerous sub-categories. The representativeness and quality depends on the authors of the articles and the collective reviews and changes. We are going to access the articles through the online-platform 'KnowledgeStore'(https://knowledgestore.fbk.eu).

### The User Profile

For the user profiles, we considered two different approaches. The first approach is to build new user profiles in comparison to other user profiles and the second one is to build every new user profile from scratch, merely depending on the own preferences and behaviour. In order to base user profiles on other profiles, we first need to produce the basis of user profiles. To generate such, we have talked about two possibilities with the other groups. Either every group member manually develops a certain number of user profiles by manually liking or disliking random articles or we automatically generate different profiles by automatically liking or disliking random articles. Liking or disliking articles means that the we save the article with its associated categories and annotate it either with a 1 for liking or a 0 for disliking. The resulting array would look somewhat like this: user_X: [[Article_1, 0], [Article_2, 1], … , [Article_15, 1]]. We can then determine articles a user with the produced profile is likely to like, when recommending articles associated with categories of articles that have previously been liked. The problem with manually producing user profiles is that every member must have a certain person in mind and rank the articles as the person would, or we would end up with user profiles only based on our personal interests and therefore with a possibility of meaningless user profiles. Either way, the profiles are probably not going to cover every new user’s interest pattern. The latter possibility of automatically liking or disliking articles is very likely going to end up with meaningless user profiles as well.   

Based on this consideration, we decided that if we cannot get a very good existing dataset of user profiles for news articles, it is not going to help to use other user profiles to compare new users with. Accordingly, we will build a user profile for new users from scratch and can consider using the so produced profiles at a much later stage, when sufficient data is available, to compare new users with. In order to build a user profiles from scratch, we considered the possibility of starting off with a questionnaire. Of course, it would be great if the user did not have to go through such a process, but it seems to be the most straightforward way to get information about the user’s interests. If we manage to build a questionnaire that is flexible enough, we will be able to only ask relevant questions without boring the user and still give good recommendations right from the beginning. As an alternative or an addition, we will consider explicit user feedback, asking the user whether he/she liked or disliked the article he/she just read. A more elegant way of tracking the user’s behaviour would be to measure how long the user spend reading the article, because if it was only a few seconds, the article was probably not very interesting for the user, but this would be out of the scope of this course.

The approach of developing a new user profile we aim for will look as follows: The new user fills out a questionnaire and based on this, an initial user profile is created. From the questionnaire, our algorithm assumes certain weights for the different categories covered. The categories are ordered in descending order of their weights. Articles with at least one of the categories with the highest weights (and none of the categories with the lowest weights) will be considered as a recommendation. Of these, more recent articles are preferred over older articles and articles that have been seen before are not going to be recommended again. The weights of the categories and the categories considered are adapted after every user feedback. How the weights are adapted is going to depend on the breadth of the category. If an article from a category that only includes 3 articles, like Figure Skating, is liked, it is likely that the other two articles are also interesting to the reader. But if an article is liked which is associated with a very large category like North America, the probability of the user to be interested in every other article in that category is very low.


## Documentation Part 3 -  Data Generation

In order to generate an appropriate sized set of training and test data for our classifier, we decided to generate 100 userprofiles, each user of which randomly either likes or dislikes one of the main categories. Four articles of each category are extracted from the dataset of news articles on wikinews.org and automatically annotated according to the category preferences(liked or disliked). Our classifier will then be trained on half of the extracted articles and tested on the other half of the articles. Accordingly the classifier will train on an tuple of the form (user, article, like/dislike) and will be tested on a tuple of the form (user, article, ?). After training, the classifier will be able to score the given articles and recommend articles accordingly.

The user profiles will look as follows: 

|                          |user 1|user 2|user 3|user 4|
|--------------------------|------|------|------|------|
|Crime and law             |1     |1     |0     |0     |
|Culture and entertainment |0     |1     |1     |0     |
|Disasters and accidents   |0     |0     |0     |1     | 
|Economy and business      |1     |1     |1     |1     |
|Education                 |0     |1     |0     |0     |
|Environment               |1     |0     |1     |0     |
|Health                    |0     |0     |0     |1     |
|Obituaries                |1     |0     |0     |1     |
|Politics and conflicts    |1     |1     |0     |1     |
|Science and technology    |0     |0     |1     |0     |
|Sports                    |0     |1     |0     |0     |
|Wackynews                 |0     |0     |1     |1     |
|Weather                   |1     |1     |0     |0     |

                
An article that user 1 would like would hence include one or more of the categories Crime and law, Culture and entertainment, Disasters and accidents, Health, Politics and conflicts, Science and technology, Weather. We would therefore pick four articles out of each category, so that we end up with 52 articles that are either associated with one or more of  the liked categories and are not associated with any of the disliked categories or that are associated with the disliked and with none of the liked articles

For the random assignments of likes and dislikes, we use the pandas library[2], which can randomly output a number in a given range, which will in our case be 1 or 0. To extract articles that belong to a certain category but not to certain others, we use the function get_applicable_news_categories() located in the folder knowledgestore and defined in ks.py or directly with create_data_set.py. in the folder topic_classification.

## Documentation Part 4 - Dataset Generation II

This week we have started to generate our dataset. We decided to look at the size of the categories provided on wikinews.org and include all (meaningful) categories which contain more than 500 articles. We picked and filtered the categories by hand, so that categories like wikinews-users who wrote and published more than 500 articles were excluded from the list of categories. We ended up with the following list of categories:

#### All Categories:
'Africa', 'Asia', 'Australia', 'Aviation', 'California', 'China', 'Computing', 'Crime and law', 'Culture and entertainment', 'Disasters and accidents', 'Economy and business', 'Elections', 'England', 'Environment', 'Europe', 'France', 'Health', 'Human rights', 'India', 'Internet', 'Iraq', 'Israel', 'London', 'Middle East', 'New Zealand', 'North America', 'Obituaries', 'Oceania', 'Politics and conflicts', 'Religion', 'Russia', 'Science and technology','Space',  'Sports', 'Transport', 'United Kingdom', 'United Nations', 'Wackynews', 'Weather', 'World'
 
For each of these categories, we will save 100 random articles from the resourceURIs.pickle file in another pickle file (see article_collection.py). These articles are going to be annotated with a 1 for like or a 0 for dislike, depending on the user liking or disliking the category it represents. 

Some of the extracted categories are at the same time subcategories of other categories, so we decided to put them in an hierarchical order and ended up with the following lists: 

#### Top-Level Categories: 

“Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business", "Environment", "Health", "Science and technology", "Sports", "Wackynews", "Weather", "Politics and conflicts", "Obituaries", "Transport", "World", "Internet", "Religion"

#### Region Categories: 

"Africa", "Asia", "Europe", "Middle East", "North America", "Oceania"

#### Subcategories:

Transport: "Aviation"                            

Science: "Computing", "Space"

Politics: "Elections", "Human rights", "United Nations"

Asia: "China", "India", "Russia"

Middle East: "Iraq", "Israel"

Oceania: “Australia", "New Zealand"

North America: "California"

Europe: "United Kingdom", "France"

UK: "England"

England: "London"


We assigned the subcategories by hand, because they were sometimes overlapping or in our eyes not very meaningful in wikinews.org. Also, even though categories may have subcategories in wikinews.org, the subcategories can be assigned to an article independently of its superordinate category. For example an article can belong to the category “London”, without being assigned to the category “England” or “UK”. Still, we wanted to include sub- and superordinate categories relations, because we think that it is unlikely that a person that is not interested in articles about “Politics” will be interested in articles about “Elections”. We therefore thought about how to integrate this constraint and ended up with the following solution: 

As described in the last section, for each user that we generate for our dataset, we will randomly assign if he or she likes each of the top-level categories or not. If a top-level category is liked, each of its subcategories will randomly be assigned a like or dislike, whereas if a top-level category is disliked, each of its subcategories will be assigned a dislike as well. Finally, the user will be named according to the categories that have been liked: e.g. username: Crime_and_law_Economy_and_business_Environmen_Health_Internet_Religion


### Current State of Achievements:

We have made a list of 100 articles for each of the top-categories and saved them into a pickle file (user_articles.pickle). The code we wrote for this procedure (article_collection.py) had some problems when accessing the wikines.org website, which is why we have only collected articles for the top-level categories and not for all of the categories yet. We will still need to collect articles for the Region categories and the subcategories. Also, we will generate the user profiles as described above in user_generaton.py and will create the final dataset by automatically annotating the articles according to each user’s preferences (dataset_generation.py and dataset.pickle)  and splitting the dataset  into 80% training and 20% test-set. We will upload the files user_generation.py, dataset_generation.py and dataset.pickle as soon as they are up and running. 

## Documentation Part 5 - Dataset Generation III

This week we have further optimised our code to extract 100 articles per category from wikinews.org and save them in a pickle file. We have found a few more categories, ‘South America’, ‘Football (soccer)’, ‘Germany’, ‘Canada’, ‘United States’, which we had overlooked in the first run, so that our lists of categories and subcategories look as follows: 

#### All Categories: 
"Crime and law", "Culture and entertainment", "Disasters and accidents",  "Economy and business", "Environment", "Health", "Science and technology", "Sports", "Wackynews", "Weather", "Politics and conflicts","Obituaries", "Transport", "World", "Internet", "Religion", "Africa", "Asia", "Europe", "Middle East", "North America", "Oceania", "Aviation", "Computing", "Space", "Elections", "Human rights", "United Nations", "Football (soccer)", "China", "India", "Russia", "Iraq", "Israel", "Australia", "New Zealand", "Canada", "United States", "California", "United Kingdom", "France", "Germany", "England", "London", “South America”

#### Top-Level Categories: 

“Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business", "Environment", "Health", "Science and technology", "Sports", "Wackynews", "Weather", "Politics and conflicts", "Obituaries", "Transport", "World", "Internet", "Religion"

#### Region Categories: 

"Africa", "Asia", "Europe", "Middle East", "North America", "Oceania", “South America”

#### Subcategories:

Transport: "Aviation"                            

Science: "Computing", "Space"

Politics: "Elections", "Human rights", "United Nations"

Sports: “Football (soccer)”

Asia: "China", "India", "Russia"

Middle East: "Iraq", "Israel"

Oceania: “Australia", "New Zealand"

North America: “Canada”, “United States” 

USA: "California"

Europe: "France", “Germany”, "United Kingdom"

UK: "England"

England: "London"

We have uploaded the code for unser generation (user_generation.py) and article extraction (article_collection.py) and also the pickle file (user_articles.picke) that contains 100 articles per category is uploaded. 
At the moment we are thinking about an elegant way to implement that for each top category that is disliked by the user (i.e. annotated with “0”), all subcategories are also set to dislike/ “0”. When we have solved that problem, the last step will be to extract the articles’ name and content text and annotate the articles as being liked or disliked by the according users’ preferences. The articles’ text will be made available to the classifier, but not the articles’ categories. Hence, the classifier will receive the following information for training: 

- article’s name 
- article’s text
- user name
- user likes or dislikes the article

## Documentation Part 6

### Dataset

Our dataset consists of 450,000 (45 categories * 100 articles* 100 users) entries of user and article-text, annotated with a like or dislike. It is easy for us to increase or decrease the the size of our dataset by increasing or decreasing the number of users. We could also increase or decrease the number of articles we extract for each category, but here we are limited to 500 articles, because we chose to include only categories with at least 500 articles. When running user_generation.py, the resulting dataset looks somewhat like this:  
 
|user                                          |article text                |label  |
|----------------------------------------------|----------------------------|-------|
|010100010011110101001101100100101110010101000 | category1_article1         |0      |
|010100010011110101001101100100101110010101000 | category1_article2         |0      |
|010100010011110101001101100100101110010101000 | category1...               |0      |
|010100010011110101001101100100101110010101000 | category1_article100       |0      |
|010100010011110101001101100100101110010101000 | category2_article1         |1      |
|010100010011110101001101100100101110010101000 | category2_article2         |1      |
|010100010011110101001101100100101110010101000 | category2_...              |1      |
|010100010011110101001101100100101110010101000 | category2_article100       |1      |
|010100010011110101001101100100101110010101000 | category45_article1        |0      |
|010100010011110101001101100100101110010101000 | category45_...             |0      |
|010100010011110101001101100100101110010101000 | category45_article100      |0      |
|101010000010111101001110101011101001001010111 | category1_article1.        |1      |
|101010000010111101001110101011101001001010111 | category2_…                |1      |

### Splitting the dataset

Since our dataset is quite big and has the potential to be extended, we first thought it would not be necessary to perform k-fold cross validation, but since we are curious about the potential differences in the results when splitting the dataset differently, we decided to try a 10-fold cross validation on our dataset. The dataset will therefore be split in 10 subsets of equal size containing random data-pairs of user and article plus the corresponding label. Training- and Testsets are split in a 80 - 20 ratio (see split_data.py).

### Evaluation metric(s) 

Because precision and recall are by themselves not very meaningful, we want to use the F1-score as our first evaluation metric, since it is the harmonic mean of the two values. Also, we want to use Cohen’s Kappa, because it is a more complex and meaningful metric than the  mere accuracy of our classifier. 

### Baselines 

The baselines we are going to use are the same as the ones we looked at in class, namely “Always True”, “Always False”, “50-50” and “Label Frequency”. Our dataset contains a bit more examples of articles the user is not interested in than articles they are interested in which has the following reason: In the first step, the user likes or dislikes all categories randomly equally. In the second step, the top-categories are checked and if they are annotated with a 0 (= dislike), the according subcategories are also assigned 0. If, on the other hand, a top category is annotated as 1, the subcategories are not changed, which means they can still be annotated with either 0 or 1. In the next step, the user gets paired with the articles of each category and the labels are given according to whether the category the article is associated with is liked or disliked. 
22 out of the 45 categories are subcategories and have therefore a probability which is double as high to be 0 as for the other 23 categories. Therefore, the probability of a user article combination to be 0 is (0.5*23+0.75*22) / 45 = 0.62 and for a user article combination to be annotated as 1: (0.5*23+0.25*22) / 45 = 0.38 .

number of positive examples: 450,000*0.38 = 171.000

number of negative examples: 450.000*0.62= 279.000


|                    |True Positive    | False Positive     | False Negative    |True Negative     |
|--------------------|-----------------|--------------------|-------------------|------------------|
|Always True         |  171,000        |  279,000           |  0                |  0               |
|Always False        |  0              |  0                 |  171,000          |  279,000         |
|50-50               |  85,500         |  175,500           |  85,500           |  175,500         |
|Label Frequency.    |  64,980         |  123,120           |  123,120          |  252,720         |

### Performance of Baselines: 

|                    |Always True      | Always False   | 50-50          |Label Frequency |
|--------------------|-----------------|----------------|----------------|----------------|
|Accuracy            |  0.38           |  0.62          |  0.5           |  0.706         |
|Precision           |  0.38           |  0             |  0.33          |  0.35          |
|Recall              |  1              |  0             |  0.5           |  0.35          |
|Cohen’s Kappa       |  0              |  0             |  -0.19         |  -0.022        |

## Documentation Part 7

#### Feature Selection: 
We are going to extract the main keywords of a given article through TF-IDF. We are going to save the most relevant (maybe 5) keywords as potential user preferences and annotate them with a weight according to whether the word appears merely in liked articles, merely in disliked articles or in both. Also, the TF-IDF score may influence the weight of the keyword. Extracting this kind of feature may be useful, because it tells the classifier something about the user’s preferences according to the keywords in the liked and disliked articles. Having this, new articles can be easily categorized in rather liked or rather disliked by the according user. 

Furthermore, we thought about using WordNet and extracting synonyms, close hyponyms and close hypernyms of the extracted keywords as even more potential keywords with similar weights ascribed to them (e.g. ‘Earthquake’ is a keyword extracted through TF-IDF, a hypernym may be ‘natural disaster’).  Also, we thought about using dbpnet to look up the class a keyword is an Entity of (e.g. ‘Trump’ is an Entity of ‘Politician’) and use the class as another keyword. We are not quite sure about how both of these ideas will work out, but they are worth to think about more in the next week.

As a second feature, we want to use word embeddings and especially word2vec to calculate the ‘semantic value’ of each word and subsequently of the whole article. We are going to use the word2vec provided by Google (see https://code.google.com/archive/p/word2vec/ ), which takes a text as an input and produces word vectors as an output. These vectors can then be represented in a semantic vector-space. By adding all the values of the word vectors and taking the mean, we yield a (rather abstract but potentially useful) semantic vector of the whole article. We can use these vectors to calculate the semantic similarity of different articles by taking the cosine distance. We can use these features for positive and negative examples equally and it is helpful, because it helps to classify new articles efficiently. 

# Documentation Part 8: 

## Feature Extraction:

The folder “data” contains all pickle files with the data we use and the keywords.txt file, containing a list of ten keywords for each category. The class feature_extraction.py takes the article and the category name and contains all methods used for feature extraction. The file user_generation.py uses the methods from freature_extraction.py to extract the different features from our dataset. We have completed to extract feature 2 and 4. The extraction of feature 3 is as good as done, but for the final feature extraction for feature 1, we still have a question for Lucas. 

### Feature 1: 
As a first feature, we measured the “semantic similarity” between an article and each category. In the first step, we extracted all words from an article, cleaned the word list from stopwords, e.g. prepositions and articles, with the english nltk stopwords list, and words that occur more than once. Also words that are shorter than four characters like “was”, “is” or “can” are excluded, if not an acronym, because they most likely carry no meaning. Afterwards, we looked up each of the resulting word’s value in the word2vec provided by Google (see https://code.google.com/archive/p/word2vec/ ). We saved the resulting 300-D vector for each word, so that we could calculate the mean vector of all the words, yielding a representation of the whole article. We also looked up the vectors for our categories (if they consist of two or more words, again, we took the mean) and calculated the cosine similarity between the article’s vector and each category’s vector. As an exception, for the category “Wackynews”, we looked up the word2vec for “wacky” and for “news”and took the mean, because there was no representation of the word “wackynews”. Our resulting first feature is a list of numbers between 0 and 1, representing the similarity between a category and an article. For a visualization of the processes see Figure 1. The resulting pre-feature looks something like follows: 
[0.9, 0.2, 0.5, …, 0.1, 0.6, 0.8]
Out of these values, we then filtered only those, that are representing the similarity between the user and the article in question. From these, we then selected the maximum, the minimum and the mean, which are the final features we will pass to the classifier. The features look something like follows:
[0.9, 0.5, 0.7]


![Alt text](/Feature1.001.jpeg "Figure 1")


### Feature 2:

To get a second feature, we are simply going to check, whether the category name is present in the article. If the category-name contains more than one word (besides “and”), it is checked whether either word is present in the article. As an exception, for the category “Wackynews”, we checked whether either the word “wacky” or “news” is present in the article. The resulting feature is an array with 45 entries of 0 (for: not in article) and 1 (for: in article), that will look something like follows: 
[0, 0, 1, 1, ..., 0, 1, 1, 0] 

### Feature 3:

The third feature we have extracted is similar to feature 2, but instead of checking whether the category-names are mentioned in the article, we checked, whether one of the keywords representing the categories, is present in the article. As keywords, we mostly used the 10 largest subcategories or terms that are in our opinion closest related to the category, so that they best represent the category. For each keyword that is present, the value of the feature is raised by 1. For keywords that consist of more than one word, it suffices if only one of the words is present in the text to raise the value of the feature for the according category. The resulting feature will be an array of 45 numbers between 0 an the number of words in the article (if all words are keywords) and will look something like the following array: 
[0, 5, 88, 0, …, 1, 0, 0, 54]

At the moment, we have completed the list of keywords (see keywords.txt), but have not completely implemented the counter-method. We will probably be done with that by tuesday. 

### Feature 4:

The fourth feature we extracted is simply the length of the article as a single number. This feature is not saved as a pickle, because its extraction takes as much as no time. 

### Feature 5 (possibly): 

For a fifth feature, we have not yet decided if we want to implement it. It would be extracted through a similar process as for feature 1, but instead of comparing the word2vecs of the different categories with the word2vecs of the article, we compare the mean of the word2vecs of the list of keywords for each category with the word2vec of the article. Keywords can, like categories, consist of more than one word. In these cases, again the mean of the word2vecs of each word would be used. The resulting pre-feature would look something like the following array: 
[0.2, 0.0, 0.7, …, 0.2, 0.8, 0.9]
As for feature 1, if we will implement this feature, we will filter the feature set, so that we pick the maximum, minimum and mean of the features that represent the similarities of the keywords representing the categories the user liked. The resulting feature will therefore look something like the following array: 
[0.8, 0.2, 0.5]


## Sources: 
[1] Gopidi, S. T. R. (2015). Automatic User Profile Construction for a Personalized News Recommender System Using Twitter.

[2] https://pandas.pydata.org
