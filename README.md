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

In the content-based filtering approach, the descriptive attributes of items are used to make recommendations. The goal is to 
recommend items to the user that are similar to those that a user liked in the past. So at its core, content-based filtering is based on the availability of (manually created or automatically extracted) item descriptions and a profile that assigns importance to these characteristics. An Advantage of this approach is that it does not require large user groups to achieve reasonable recommendation accuracy. Moreover, new items can be immediately recommended once item attributes are available. In some domains these item descriptions can be automatically extracted or are already available in an electronic catalog. In other domains some of these characteristics are hard to acquire automatically. In that case such information must be manually entered into the system.

Both approaches can be combined into a hybrid approach, which then could be more effective in some cases. But due to a lack of user information, we would have to enter all the user data manually to use elements of a collaborative filtering approach. Given the established rules of statictics and data science, a tremendous amount of fake data would be needed for finding such intrinsic relations. As an alternative, in the content-based filtering approach, we can use SPARQL to extract automatically the pre-annotated characteristics of our news articles. Only the preferences of the user have to be simulated from us manually.

### Classifier design
Given the complexity of the task, we decided to stick to the content-based filtering approach in the beginning. In order of implementing such a news recommendation system utilized supervised machine learning, we have to define our goals. In a nutshell, we define our task as approximating an unknown function *f*, which gets the preferences of a user and an article as input and returns a score indicating the predicted interest of the user for that article. Iterating through the set of news articles and evaluating the function multiple times, such an implementation would find the most relevant articles from a corpus.  

For acquiring the required data, we have to simulate some user with specific preferences. This data will be utilized both as training and testing data. In a next step, the viewed articles will be used to extract a representation for the preferences. This preferences will afterward be used to train the second classifier mentioned above.

## Dataset (Week 2)

Given the aims of the project and its design as a supervised classification task, the data as input is a crucial past of the system. In the case of a recommendation task, the preferences of the user may be considered as the crutial infomation. In general, we might have two difference ways of representing those, independently if we want to focus on collaborative or content-based approaches mentioned above:
- In a "direct measurement", we measure the preferences of the users by asking for their opinion towards the items. The actual modality of this process are not specified. Therefore, we may use setups like binary like-dislike questions or grading scales. While the direct communication and instruction with the participants may result in high quality data, the approach scales not well. User may have to spend their time actively and need therefore to be recruited.
- "Indirect measurement" take an alternative approach: By observing the behavior of users in their normal environment, one may be able to draws clue. Especial in the contexts of websites, that is a commonly utilized approach. By using indicators like duration of stay or interacting behavior, the preference may be concluded. Such an approach scales extremely well with the number of users who may not even detect they are currently tracked. On the other side, such a approach adds a hugh number of noise due to its uncontrolled nature: A user may abort reading a newsletter article because she or he is distracted or the current state of mood may not match the general preferences.

Both approaches have advantages and disadvantages which have to be considered during the plan of the experiments. Like commonly in data science, it is the classical differentiation between small amounts of high quality data and intense amounts of rather noisy data where both may come from a specific, commonly Gaussian distribution. In this project, we will stick to the firstly formulated type of measurement due to difference reasons.

- Low number of user -> not an existing website, but from scratch
  - Easy generation of artificial data -> Independent of real data
- No focus on noise, but on pure data -> generalization later

## Citations
Aggarwal. (2016). Recommender Systems: The Textbook. Springer

Jannach, Zanker, Felfernig, Friedrich. (2011). Recommender Systems: An Introduction. Cambridge University Press

Ricci, Francesco & Rokach, Lior & Shapira, Bracha. (2010). Recommender Systems Handbook. Springer

