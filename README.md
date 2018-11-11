# ML4NLP - Alpha
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Alpha.



## Session 2, 30.10.18

### Project goal

Goal of the project is the recommendation of news articles on the basis of Wikinews (https://en.wikinews.org/wiki/Main_Page), accessed via KnowledgeStore API (http://knowledgestore2.fbk.eu/nwr/wikinews/ui). The latter includes the enrichment of articles with information in a format that makes them easiy to process by computers, for example the linking of mentioned entities with DBpedia-entries (https://wiki.dbpedia.org/).

One main approach to solve the task of news recommendation is based on collaborative filtering, another one is based on the matching of a user's profile to an article. The former approach relies on a database of preferrably many users and their interests in articles, the latter doesn't require the knowledge about other users after training. As neither Wikinews nor the additional information from the KnowledgeStore database include this user data, our approach sticks to the direct matching of a user profile to an article.

The recommendation should be able to judge if a given article matches a specific user's topics of interest. A user's topics of interest are determined by asking him or her which predefined topics match his or her interest. This profile of a user is stored as a vector in a space where each dimension is one topic of interest.

In order to estimate a user's interest in a given article, machine learning techniques are used to train a classifier. The training is done with pre-collected example data of user profiles and their interest in a number of articles.
After training, the classifier is supposed to receive one user profile and one article as an input, the output should be the statement if that article is interesting for the specific user or if not.

By having a user profile, the classifier should be able to recommend each user individual articles rather than the general recommendation of (for example popular) articles to unknown users.



## Session 3, 06.11.18

### The data collection process

In order to build a machine learning model, it is crucial to know which form the input data will have. We encountered the problem that there is no pre-existing user data that is tailored to our research problem; the data would be required to be restricted only to articles that are listed in Wikinews via the KnowledgeStore database. We collected several options to resolve this problem:

Option 1: 
We could simply assume that users can create profiles in which they specify the topics they are interested in. If the topics are broad (e.g. politics, science), the task would be simply extracting articles of a certain topic category from the database. As this would disregard the machine learning aspect, we would prefer not to use this approach.

Option 2: 
In order to incorporate machine learning techniques, we could ignore the information on the article‘s topic categories that is specified in the database and just use the raw text of the article. This would allow us to auto-generate the dataset, but would transform the news recommendation task into a topic classification task.

Option 3: 
Create several artificial user profiles as some kind of expert labeling. Each artificial user would only be interested in one rather narrow topic (e.g. a certain celebrity, a certain football team or a certain science sector). However, the topic would not explicitely be named in the profile, but the profile would consist of a list of articles on that topic. In order to generate training data, we would manually search for news articles that talk about that specific topic and thus create user profiles that consist of a list of articles that the imaginary user liked. The goal would then be to recommend articles that are about that specific topic.
Creating over-simplified profiles has the advantage that one can more easily tell whether the output of the recommender system is appropriate. On the other hand, we would not be able to tell whether our model would be able to cope with the complexity of real users‘ preference patterns. Therefore the dataset would not satisfy the desiderata of being representative.

Option 4:
Each team member would generate data for a user profile that stands for his or her own interest, thus being also an expert labeling approach. The definition of a user profile then could happen on the basis of a number of liked articles. The quality of the dataset would be high as the user profiles would match real users with the downside of being of marginally little size, even if we shared the dataset with all groups working on news recommendation.

Option 5:
Use a pre-existing dataset that contains user data, such as a list of articles they liked. This would also allow for a collaborative filtering approach. The problem here would be that a custom dataset is probably not confined on news articles that are also listed in the KnowledgeStore database, wherefore we could not make use of the meta-information linked to the articles.
The advantage would be that the recommender system would work with real data instead of artificially created data.

We have not yet decided on which option we will realize because we would like to discuss it with the lecturer or in class.
