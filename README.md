# ML4NLP
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Beta.

# Documentation Part 1: 

# Introduction
Today’s world changes at a rapid rate and we are flooded by news constantly. News apps update the user’s feed every minute and keeping up-to-date with oneself is time-consuming challenge. To reduce the amount of information to the news the user is interested in, a News Recommendation System is required. It would filter the articles to display solely the relevant reports. The News Recommendation System that will be implemented here, is going to filter news articles from the website wikinews.org.

Building a News Recommendation System, there are two common approaches: collaborative and content-based approaches (Gopidi, 2015). Collaborative or popularity-based approaches use the preferences of many users and similarity measures to predict whether or not a user will like a news article. Problems with this method arise mainly due to a lack of data. Furthermore, it is hard to make predictions for new users, because finding article-user pairs for each user is computationally complex. Also, it is very helpful to have user feedback or ratings on articles to collect information about collective preferences, which wikinews.org does not provide. 

# Approaches
Content-based approaches take into account the similarity of the content of the news article and the content of the news articles the user previously liked. The content can be either represented by keywords or the categories an article is assigned to. In wikinews.org, every article belongs to one or more categories (which often also seem to be keywords).

On the other hand, a hybrid approach could use existing user profiles that are based on collective user action. The user profile built by interpreting explicitly (questionnaire) or implicitly (tracing user activities) collected data can then be compared with the different user profiles from the collaborative approach and the user preferences of the (collective) profile closest to the user’s profile can additionally be taken into account to achieve better performance. If we can manage to obtain such collective user profiles on news, we will try to include it.

Because of the problems that come with the collaborative approach, we decided to implement a content-based news recommender system or a hybrid approach.

# Structure of our Approach (cf. Gopidi, 2015): 
1) Categories: 

Wikinews.org provides the following categories: 

Region: Africa • Asia • South / Central / North America • Europe • Middle East • Oceania • Antarctica

Topic: Crime and law • Culture and entertainment • Disasters and accidents • Economy and business • Education • Environment • Health • Obituaries • Politics and conflicts • Science and technology • Sports • Wackynews • Weather

We still need to decide on the scope of our recommender system and whether or not we want to include all categories, only the main categories, or only the subcategories of one main category, e.g. sports or economy

2) Profile Building:

We decided to collect data explicitly by collecting user feedback of every new user. There are different possibilities to collect the data. One way is to show the user several articles from different categories, which are either randomly selected or explicitly chosen to be representative for a certain category, and let the user vote on them without showing the associated categories. Another possibility is to let the user directly vote on the categories in question. 

It may be possible to let the user choose which regional scope he/she is interested in regarding the different topics, e.g. Sports - worldwide, Crime and Law - Europe and Middle-East, Economy - Asia, Europe and North America, Weather - Berlin, etc. 
Also, when the user expresses an interest in Sports, it may be useful to further ask in what kind of sports (wikinews.org provides 59 subcategories), whereas for other categories the subcategories are less explicitly stated and should therefore not be queried.

3) News Recommendation:

In the final step, we will give the recommender the user’s profile and the categorized articles as an input and receive recommendations based on their similarity as an output.


# Sources: 
Gopidi, S. T. R. (2015). Automatic User Profile Construction for a Personalized News Recommender System Using Twitter.

