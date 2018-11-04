# ML4NLP - Alpha
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Alpha.

## Session 2, 30.10.18

### Project goal

Goal of the project is the recommendation of news articles on the basis of Wikinews (https://en.wikinews.org/wiki/Main_Page), accessed via KnowledgeStore API (http://knowledgestore2.fbk.eu/nwr/wikinews/ui). The latter includes the enrichment of articles with information in a format that makes them easiy to process by computers, for example the linking of mentioned entities with DBpedia-entries (https://wiki.dbpedia.org/).

The recommendation should be able to judge if a given article matches a specific user's topics of interest. A user's topics of interest are determined by asking him or her which predefined topics match his or her interest. This profile of a user is stored as a vector in a space where each dimension is one topic of interest.

In order to estimate a user's interest in a given article, machine learning techniques are used to train a classifier. The training is done with pre-collected example data of user profiles and their interest in a number of articles.
After training, the classifier is supposed to receive one user profile and one article as an input, the output should be the statement if that article is interesting for the specific user or if not.

By having a user profile, the classifier should be able to recommend each user individual articles rather than the general recommendation of (for example popular) articles to unknown users.
