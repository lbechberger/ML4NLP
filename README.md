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

* Option 1: 
We could simply assume that users can create profiles in which they specify the topics they are interested in. If the topics are broad (e.g. politics, science), the task would be simply extracting articles of a certain topic category from the database. As this would disregard the machine learning aspect, we would prefer not to use this approach.

* Option 2: 
In order to incorporate machine learning techniques, we could ignore the information on the article‘s topic categories that is specified in the database and just use the raw text of the article. This would allow us to auto-generate the dataset, but would transform the news recommendation task into a topic classification task.

* Option 3: 
Create several artificial user profiles as some kind of expert labeling. Each artificial user would only be interested in one rather narrow topic (e.g. a certain celebrity, a certain football team or a certain science sector). However, the topic would not explicitely be named in the profile, but the profile would consist of a list of articles on that topic. In order to generate training data, we would manually search for news articles that talk about that specific topic and thus create user profiles that consist of a list of articles that the imaginary user liked. The goal would then be to recommend articles that are about that specific topic.
Creating over-simplified profiles has the advantage that one can more easily tell whether the output of the recommender system is appropriate. On the other hand, we would not be able to tell whether our model would be able to cope with the complexity of real users‘ preference patterns. Therefore the dataset would not satisfy the desiderata of being representative.

* Option 4:
Each team member would generate data for a user profile that stands for his or her own interest, thus being also an expert labeling approach. The definition of a user profile then could happen on the basis of a number of liked articles. The quality of the dataset would be high as the user profiles would match real users with the downside of being of marginally little size, even if we shared the dataset with all groups working on news recommendation.

* Option 5:
Use a pre-existing dataset that contains user data, such as a list of articles they liked. This would also allow for a collaborative filtering approach. The problem here would be that a custom dataset is probably not confined on news articles that are also listed in the KnowledgeStore database, wherefore we could not make use of the meta-information linked to the articles.
The advantage would be that the recommender system would work with real data instead of artificially created data.

We have not yet decided on which option we will realize because we would like to discuss it with the lecturer or in class.


## Session 4, 13.11.18

### Decision how to aquire the data set

The decision how to aquire a data set resulted in the plan to auto-generate idealized user profiles, so option 3 of the preceding part of the documentation.
Before going further in this direction, the reasons for not taking the other options are shortly explained.

* Option 1 doesn't include machine learning, so the seminar's goal wouldn't be covered.

* Option 2 is in fact topic classification rather than news recommendation. 

* Taking option 4 doesn't result in a sufficiently large data set. The only reasonable possibility to get enough data would have been the agreement of all three News Recommendation groups to take this approach and the sharing of data sets.

* Option 5 would not include the use of the KnowledgeStore database to create the data set which doesn't match the seminar's criteria. Furthermore, a look at a fitting dataset by Yahoo revealed that the handling of such would go beyond the seminar's scope.

### User profiles

By chosing the automatic generation of idealized user profiles to build the data set, the way to model a user's profile is also narrowed down. As described in the preceding part of the documentation, a user profile from the point of view of the classifier is just a number of articels that match the user's interests. To give an example, these articles could be articles which the user read til the end.

### Aquiring the data set

However, in order to aquire the data set, an intermediate step to model user's interests is taken. Interests are defined by a small number of topics the user is interested in. To enable the automatic generation, the topics of interest match the categories into which the articles of Wikinews are sorted. As the amount of top level categories is low at 16, we use the much bigger number of subcategories.
A user profile consists of number (still to be defined) of articles that belong into the topics of interest for a user. In order to generate the user profiles, we created a python dictionary that lists for each topic all articles that fall into that topic. The articles are drawn from a random distribution where each topic is weighted by the number of articles that belong to that topic. We chose this approach because we assume that topics that contain a large number of articles are more important as more people are interested in these topics.



## Session 5, 20.11.18

During this week, we finished the creation of the dataset. *dataset_generation.py* holds the method *generate_dataset(amount_users, subcategories_per_user, articles_per_user_and_category)* which creates a desired number of user profiles, each one consisting of a defined number of categories of interests per user and a defined number of articles per user's interest. This ensures a uniform format in which each user profile is defined by an equal amount of articles the user liked (the user's categories of interest are not specified in the dataset). The user's categories of interests are drawn from a weighted random distribution where categories that contain a larger number of articles are more likely to be drawn than categories containing a smaller number of articles. This decision was made because we argued that, in general, a category that contains many articles is more important and more people are interested in that topic. Subsequently, for each of the user's topics of interest, a specified number of articles from that category are randomly drawn.
The dataset is returned as a list of URIs and can be saved e.g. in a csv-file.

Topics of interest are chosen from the category mapping of Wikinews. Note that the top-level categories aren't taken into account but only the level-2-categories, so subcategories. We excluded the first ones because if such a category was chosen for a user's topic of interest, the thematic range of articles belongig to that category would be high compared to articles belonging to a subcategory.

The python code for generating users uses the method *create_category_articles_dictionary()*, for which we extended *ks.py* in the *knowledgestore* folder. *create_category_articles_dictionary()* creates a dictionary that matches news articles to the subcategories to which they belong. In order to get the subcategories for an article, the method *get_all_news_subcategories(resource_uri)* looks at the HTML code of the corresponding Wikinews article website, given by the parameter *resource_uri*. The HTML code contains the string *wgCategories*, which is followed by a list of categories that article belongs to. The top-level categories are excluded as explained above. 
The extraction of categories for every article took roughly two hours, so the resulting dictionary is stored in the file *subcategory_resource_mappings.pickle*. If *create_category_articles_dictionary()* finds that file, it loads the dictionary from there instead of creating it anew.


## Session 6, 27.11.18

In this week we finished creating the dataset. The code for generating the dataset can be found in the file *generate_dataset.py*. 
The dataset consists of 1000 user profiles. Each user profile contains 30 articles the user liked - 10 from each of the 3 topics the user is interested in.
Apart from that, there are also 30 positive and 30 negative training examples per user which can later be used for training the classifier.

The data is saved in a nesting of lists, which have the following hierarchical structure (example for two users):

<pre>
[                                                       dataset                                  ]

[ [                         user1                                 ] , [         user2          ] ]

[ [ [   profile    ] , [               training                 ] ] , [ [profile] , [training] ] ]

[ [ [liked articles] , [ [liked articles] , [disliked articles] ] ] ,           ...              ]
</pre>
    
The *generate_dataset.py* program does the following:
Firstly, a dictionary with a matching from all subcategories to a list of all articles that belong to the specific subcategory is created. Afterward, certain categories are deleted:
* all categories that contain too few articles (less than 15 or less than twice as many as the variable that denotes the number of articles that are chosen in the user profile per topic
* all categories that contain too many articles (over 506). We argued that categories that are very large are too general and hence the articles from that category do not have much in common.
* all categories that denote a specific date, e.g. 'January 1, 2008'. 
* all categories that just describe authorship. For example 'Cocoaguy (Wikinewsie)' or 'Juliancolton (WWC2010)'. The reason for that is that some authors write about a wide, seemingly unrelated variety of articles, hence they do not have anything to do with a certain topic.
* the following categories: 
  * 'Published'
  * 'Archived'
  * 'Original reporting'
  * 'AutoArchived'
  * 'Pages with template loops'
  * 'Pages using duplicate arguments in template calls'
  * 'Pages with pull-quotes'
  * 'Pages with defaulting non-local links'
  * 'Pages with categorizable local links'
  * 'Pages using two-parameter languageicon'
  * ''
  * 'Pages with missing-image template calls'
  * 'Pages with forced foreign links'
  * 'Pages using three-parameter languageicon'
  * 'Reviewed articles'
  * 'Pages with irredeemable missing-image template calls'
  * 'Corrected articles'
  * 'Writing contest 2010'
  * 'Imported news'
  * 'Translated news'
  * 'Featured article'
  * 'Writing Contests/May 2010'
  * 'News articles with translated quotes'
  * 'News articles with telephone numbers'
   
After deleting these unsuitable categories, we used a weighted random distribution to chose three distinct topics of interest for each user. The more articles a category has, the more likely it was chosen. 
Then, for each topic of interest, there were 20 articles drawn - 10 for the user profile and 10 for the positive training samples. Afterward, 30 articles that do not belong to any of the three categories of 
interest were drawn in order to be used as negative training samples.
Finally, the dataset was saved as a pickle file.

## Session 7, 04.12.18

### Splitting up the dataset

As the dataset is auto-generated, it is big (theoretically limited only by the size of Wikinews/amount of article there), so the use of cross-validation doesn't seem necessary for creating our classifier. The same argument counts against the usage of the same data for training, test and validation. As there is a lot of data present in the set, we can use different parts of the set for training, test and validation.
Nevertheless, the division of the dataset needs additional consideration. In class, we discussed the example of splitting up the dataset for summarization: there should be some articles that are not known for the classifier during training. We want the same for our classifier for news recommendation. One solution that comes into mind is the reservation of articles that a newer than a certain date and putting them aside for test and validation data.

### Evaluating the classifier's performance

For evaluating the classifier's performance, we want to use several metrics. As Precision and Recall aren't that meaningful for themselves, we want to use the F-score as a combination. A decision still to be made is if we use the balanced F1-score or the F2-score to weigth recall higher than precision. The reason is that for each user the number of positive examples is much lower than the amount of negative ones. Wherefore the error of not recommending an article that would be interesting to the user is more severe than the error of recommending articles that are not interesting.
Two other metrics that we want to use are Matthews correlation coefficient and Kohen's Kappa. Being somewhat similar, both are appropriate scores for evaluating the classifiers performance.
At last, the accuracy should be calculated for having a metric that is widely used.

### Baselines

Currently, the users of our dataset have the same number of articles in which they are interested and in which they are not. Feeding the whole data for training into our classifier could result in an unrealistic bias to classify more articles as interesting for the user than it would be the case in practice. We consider changing the relation between the number of positive and negative examples in the dataset.

As baselines we are planning to use *always true, always false, 50-50, label frequency* as suggested during class. The resulting metrics for these baselines are as follows:

|  | Always “True” | Always “False” | 50-50 | Label Frequency |
|-----------------------|---------------|----------------|-------|-----------------|
| Accuracy | 0.5 | 0.5 | 0.5 | 0.5 |
| F1-Score | 0.67 | 0 | 0.5 | 0.5 |
| F2-Score | 0.8333 | 0 | 0.5 | 0.5 |
| Matthew's correlation | 0 | 0 | 0 | 0 |
| Cohen's kappa | 0 | 0 | 0 | 0 |



## Session 7, 11.12.18


### Remarks concerning the previous documentation part:

Up tp now, we had the same amount of positive and negative examples in our dataset to train the classifier. However, the percentage of articles the user is theoretically interested in is not 50% but 3.1%. We calculated the number of 3.1% by counting the average amount of articles that belong to the user's 3 categories of interest devided by the number of all articles.


### Features

We decided to uses word embeddings as well as term frequency - inverse document frequency (tf-idf) as features.
We are planning to train the word embeddings over all articles of Wikinews. One potential problem of training the word embeddings just of the Wikinews articles that are present until a certain date is that if newer articles are published is that contain unseen words. These unseen words are not present in the word embedding and therefore the classifier is not able to process them. However, in a real life scenario, we would assume regularly trained word embeddings. In our case, a work-around could be to replace unknown words with their categories that are denoted in the dbpedia. For example, if "Trump" was an unseen word, it would be replaced by "politician".
Eventually, the sum of the embeddings of the words (or the important words according to tf-idf) can be used as a feature. As stated in *Handouts_Session_8*, the sum of the word embeddings of a document retrieves an "average meaning" of the document, wherefore we think that it might be a meaningful feature. Articles with similar meaning would accordingly show embedding vectors that have a small cosine distance to each other.


Another idea to extract features to use for the classifier is to compute the tf-idf for all words in the article to be classified and use the 5 words with the highest tf-idf value. When aiming to figure out if a new article is interesting for the user, the tf-idf values for those words in the new article are computed and summed up. This value can also be used as feature for the classifier. We chose to use these words with high tf-idf values because we think that the overall topic of an article can be summarized by the "most important" words of the specific article. An article which should be classified positive would yield a high a high sum of tf-idf values for the words that have been found earlier in the user profile, whereas the sum for a uninteresting article would be small.
