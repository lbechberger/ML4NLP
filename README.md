# ML4NLP - Gamma

## Introduction (Week 1)
News recommendation is a commonly tackled task in natural language processing. In our era of massive sources of information available, the appropriate pre-selection of candidates matching the specific interest of a user is a crucial task. Both the user and the providing platform profits from such a system, as users whose needs are fulfilled may stay longer in such an environment. Given its purpose, news recommendation may be seen as an instance of classical Recommender System. According to Ricci et al. (2010), such a system is composed out of three basic components:

- User: A user is an entity having an interest in consuming items given its special preferences.
- Item: A product is an entity from a set, which is chosen by the system for best matching the preferences of a user. In our case, these items are news articles.
- Preferences: A preference is a feeling of a user regarding an item, commonly described as likes and dislikes.

Many current websites use such systems for different modalities. Examples are the selling of products (i.e. Amazon), the recommendation of movies and series (i.e. Netflix) or the providing of somewhat pleasant, interesting or funny posting (i.e. Facebook).

### Project goal
The goal of this project is to implement a news recommendation system based on articles of the free-content news source Wikinews (https://www.wikinews.org/). As the data base we will use KnowledgeStore (https://knowledgestore.fbk.eu/), in which entities and events from Wikinews have been extracted and which is enriched with knowledge from dbpedia (https://wiki.dbpedia.org/). 

With the help of SPARQL (https://www.w3.org/TR/rdf-sparql-query/) and the KnowledgeStore API (http://knowledgestore2.fbk.eu/nwr/wikinews/ui) we will access and extract knowledge from this data base to easily gather the data for our news recommendation system in a computer readable format.

### General approaches
In order to understand the design decisions of our project, we sill shortly introduce the main idea of popular approaches for Recommender Systems. In generel two popular approaches can be distinguished. A list of recommendations can be gathered either by
applying collaborative filtering or by applying content-based filtering.

In the collaborative filtering approach, recommendations are selected by collecting and comparing information and preferences from multiple users. The underlying assumption is that if a person A has the same interest as a person B, the person A is more likely to have person B's interests than that of a randomly chosen person. Pure collaborative filtering approaches do not exploit or require any knowledge about the items themselves. An advantage of this strategy is that these data do not have to be entered into the system or maintained.

In the content-based filtering approach, the descriptive attributes of items are used to make recommendations. The goal is to recommend items to the user that are similar to those that a user liked in the past. So at its core, content-based filtering is based on the availability of (manually created or automatically extracted) item descriptions and a profile that assigns importance to these characteristics. An Advantage of this approach is that it does not require large user groups to achieve reasonable recommendation accuracy. Moreover, new items can be immediately recommended once item attributes are available. In some domains these item descriptions can be automatically extracted or are already available in an electronic catalog. In other domains some of these characteristics are hard to acquire automatically. In that case such information must be manually entered into the system.

Both approaches can be combined into a hybrid approach, which then could be more effective in some cases. But due to a lack of user information, we would have to enter all the user data manually to use elements of a collaborative filtering approach. Given the established rules of statictics and data science, a tremendous amount of fake data would be needed for finding such intrinsic relations. As an alternative, in the content-based filtering approach, we can use SPARQL to extract automatically the pre-annotated characteristics of our news articles. Only the preferences of the user have to be simulated from us manually.

### Classifier design
Given the complexity of the task, we decided to stick to the content-based filtering approach in the beginning. In order of implementing such a news recommendation system utilized supervised machine learning, we have to define our goals. In a nutshell, we define our task as approximating an unknown function *f*, which gets the preferences of a user and an article as input and returns a score indicating the predicted interest of the user for that article. Iterating through the set of news articles and evaluating the function multiple times, such an implementation would find the most relevant articles from a corpus.  

For acquiring the required data, we have to simulate some user with specific preferences. This data will be utilized both as training and testing data. In a next step, the viewed articles will be used to extract a representation for the preferences. This preferences will afterward be used to train the second classifier mentioned above.

## Dataset (Week 2)

Given the aims of the project and its design as a supervised classification task, the data as input is a crucial part of the system. In the case of a recommendation task, the preferences of the user may be considered as the significant information. In general, we might have two different ways of representing it, independently if we want to focus on collaborative or content-based approaches mentioned above:
- In a "direct measurement", we measure the preferences of the users by asking for their opinion towards the items. The actual modalities of this process are not specified. Therefore, we may use setups like binary like-dislike questions or grading scales. While the direct communication and instruction with the participants may result in high-quality data, the approach scales not well. The users may have to spend their time actively and need therefore to be recruited.
- "Indirect measurements" take an alternative approach: By observing the behavior of users in their normal environment, one may be able to draw some conclusions. Especial in the contexts of popular websites, that is a commonly utilized approach. By using indicators like duration of stay or interacting behavior, the preference of a visitor may be concluded. Such an approach scales extremely well with the number of users who may not even recognize they are currently tracked. On the other side, such an approach adds a tremendous amount of noise due to its uncontrolled nature: A user may abort reading a newsletter article because she or he is distracted or his or her current mood does not match his or her general preferences.

Both approaches have advantages and disadvantages which have to be considered during the plan of the experiments. Like commonly in data science, it is a classical weighing between small amounts of high-quality and colossal amounts of rather noisy data where both may come from a specific distribution. In this project, we will stick to the firstly formulated type of measurement due to different reasons. First of all, we have no access to an existing news website as a potential source of user data. Especial under the influence of modern data protection laws, a collection of extensive amounts of data seems not to be possible. Without having to care about the intrinsic noise of the data, one may focus more precisely on the actual goal of the experiment. In order of acquiring high-quality data, we stick therefore to explicitly annotated preferences. As the topics of interest will be rather high level in our setup, we have even the opportunity to utilize artificially generated data. Given broad and mutable exclusive categories of interest, we may simulate "ideal" user.

In order of encoding the preference in our setup, two different ways of a binary code seem reasonable. On the one hand, we may annotate both "likes" and "dislikes" regarding a set of articles. On the other hand, we may create exclusively a set only with articles of interest and considering everything outside as "not of further interest". As it may simplify later the introduction of noise while keeping the data management simple, we will stick to latter way.

Summarizing these design decisions, we utilize a hybrid approach by combining a pre-existing set of newspaper articles with artificial users with "expert-made" yet idealized preferences:

1. A collection of mutable exclusive topics of newspaper articles is defined by human experts (i.e. "Porsche car", "Mid-elections in the US").
2. Human experts assign a number of newspaper articles to each topic.
3. A massive amount of artificial users are generated by assigning different topics of interests to them. 
4. During training, a subset of the articles of each category is randomly selected and trained to result in a match regarding another article of the category. Furthermore, newspaper articles from other categories are utilized to train the binary classifier regarding a negative response.
5. For evaluation purposes, articles of each category not used in training are evaluating if being correctly classified.

This approach allows covering a large number of desirable properties of a dataset. Due to its artificial nature, a large amount of data can be generated effortlessly. The quality of the data is sound while being clearly documented. Probably, the biggest source of potential trouble is the representative nature of the data regarding the "real world". A careful selection of an extensive set of appropriate topics with assigned newspaper articles is necessary.

## Source of the dataset (Week 3)
After defining the general approach, the next step consists of evaluating which kind of dataset might be suitable for the proposed method. We need to define the exact source for our samples. Our first starting points were the annotated samples provided by the KnowledgeStore. Nevertheless, given its publication date and the commonly rather rapid development of popular web pages, we decided, that it might be worth trying to obtain more samples as the annotated ones by accessing Wikinews directly. 

Wikinews differentiates between categories and articles. One category can have zero, one or multiple subcategories. The subcategories are normal categories as well and can have zero, one or multiple subcategories on their own again. An article can be assigned to every category. Every category may have multiple subcategories and articles assigned simultaneously. The category "Music", for example, has both the subcategories "Blues music", "Classical music", "Heavy metal" etc. but also multiple articles, e. g. "US rapper Mac Miller dies at home in Los Angeles" or "Netta wins Eurovision Song Contest for Israel" assigned. Given this hierarchical structure, a representation as a tree-like structure seems suitable to represent the data.

![Hierarchy](Hierarchy.png)

For querying all the actual data, two ways seemed possible and were evaluated. The foundation for all our experiments was an object-orientated pipeline written in Python allowing a flexible parsing, filtering and storing of both categories and articles. The classes may be found in the file "ArticlesExtractor.ipynb".

- As a first approach, Wikinews offers its own page, where every category and the number of assigned news articles to each category are listed (https://en.wikinews.org/wiki/Special:Categories). Unlike one may expect, this page is automatically generated by the underlying MediaWiki-System. As an implication,  the list contains not only categories but also a high amount of noise: The pages of specific users and authors are included beside "Meta-categories" like "Corrected Articles" or "Templates".  Besides the actual parsing, a high amount of work was employed to define filters for such outliers. Utilizing more than 30 different regular expressions both for categories and articles, we defined a rather stable subset of valid entities. Nevertheless, fearing pollution and a drop in quality we do not follow this approach. 

- As a second approach, we defined a set of "top-level" categories and created the structure recursively from those. We oriented us both on the categories proposed by Wikinews and the one embedded into the KnowledgeStore. As these pages do not include any metadata, we do not apply any additional filtering.

Despite our effort, the extraction process does not lead to a potentially valuable amount of additional data. In comparison to the 19737 annotated articles in the KnowledgeStore, the 27657 articles available nowadays were not considered as justifying a complete hybrid pipeline of both sources, at least not during our first experiments. We will stick to the available categories in the KnowledgeStore in order of generating the preferences of our user.

## Generating the dataset (Week 4)

### Preprocessing
After we investing some time on the analysis of the articles and categories provided by Wikinews and the Knowledgestore in the last week, we implemented the necessary steps to generate our dataset for the classifier in this week.

First of all, we extracted every article, that exists in the Knowledgestore. Due to the huge number of articles stored in the Knowledgestore (19737 articles), this query runs a significant amount of time. For a faster access in the future, we saved all articles into a pickle-file ("articles.pickle"). 

In the next step, we obtained the corresponding categories for each article and stored them in a dictionary (function "articles_to_categories"). We make sure, that the same category is not saved multiple times in our list so that we only store the categories once. In total, we then have 5833 different categories for all articles. 

Then we applied our filter from last week onto the categories to filter out the categories, that does not make sense for our news recommendation system (e.g. "Corrected Articles" or "Audio reports"). The exact filter can be seen in the function "filter_categories".

In the next step, we looked into the distribution, how many articles are assigned to a corresponding category. [TODO: Show plot]
From this distribution, we concluded to filter out every category with less than five articles assigned (included in the function "filter_categories"). The resulting number of articles is 1612.

### Generate dataset
After that, we implemented the functions to generate the dataset (class "User"). Therefore we randomly draw two categories per user ("random.sample(categories.keys(), num_interests)") and assign all articles, that belong to these categories, to the user for an internal representation ("interests_articles").

To train our classifier, we need a user representation on the one side and the training data on the other side as an input. We obtain the user representation by drawing randomly three articles per category from the internal representation of the user. Theoretically, the categories we did not draw can be used as positive examples for our training data. For the user representation for our classifier and for the positive sample we use the function "get_positive_sample" in our source code (the user representation is stored in "input_data" and the positive sample is stored in "true_labels"). Similar to that we can draw a negative sample with the function "get_negative_sample" from our articles.

We are now able to generate positive or negative labeled training data for one single user. The idea is to generate multiple users during our training process and train our classifier on this users. Due to the combinatorics of assigning random articles to users, we can generate an incredible amount of training data and therefore we hope to get a sufficient amount of data to train our classifier on.

### Citations
Aggarwal. (2016). Recommender Systems: The Textbook. Springer

Jannach, Zanker, Felfernig, Friedrich. (2011). Recommender Systems: An Introduction. Cambridge University Press

Ricci, Francesco & Rokach, Lior & Shapira, Bracha. (2010). Recommender Systems Handbook. Springer
