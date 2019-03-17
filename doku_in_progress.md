# ML4NLP - Alpha

[Project goal](#project-goal)  
[The data collection process](#the-data-collection-process)  
[User profiles](#user-profiles)  
[Obtaining and storing subcategories](#obtaining-and-storing-subcategories)  
[Excluded categories](#excluded-categories)  
[Specification of parameters](#specification-of-parameters)  
[Splitting up the dataset](#splitting-up-the-dataset)  
[Balanced vs. imbalanced data](#balanced-vs-imbalanced-data)  
[Features](#features)  
[Feature selection / Dimensionality reduction](#feature-selection--dimensionality-reduction)  
[Scores of classifiers](#scores-of-classifiers)  
[Additional feature of shared entities](#additional-feature-of-shared-entities)  
[Classifier selection](#classifier-selection)  
[Evaluating the classifier's performance](#evaluating-the-classifiers-performance)  
[Baselines](#baselines)  
[Results](#results)  
[Missing data](#missing-data)  
[Train & Test with balanced data](#train--test-with-balanced-data)  
[Train with balanced, test with imbalanced data](#train-with-balanced-test-with-imbalanced-data)  
[Setup (How to use our code)](#setup-how-to-use-our-code)  
[References](#references)

### Project goal

Goal of the project is the recommendation of news articles on the basis of Wikinews (https://en.wikinews.org/wiki/Main_Page), accessed via KnowledgeStore API (http://knowledgestore2.fbk.eu/nwr/wikinews/ui). The latter includes the enrichment of articles with information in a format that makes them easy to process by computers, for example the linking of mentioned entities with DBpedia entries (https://wiki.dbpedia.org/).

One main approach to solve the task of news recommendation is based on collaborative filtering, another one is based on the matching of a user's profile to an article. The former approach relies on a database of preferably many users and their interests in articles, the latter doesn't require the knowledge about other users after training. The decision, which of these approaches to use for the project, is taken for the matching of a user's profile to an article. Neither Wikinews nor the additional information from the KnowledgeStore database include user data, additionally, using another dataset like the yahoo-news-dataset would be beyond the seminar's scope. Nevertheless, for training, some kind of user data is required. This topic will be revisited in the next chapter.

In order to estimate a user's interest in a given article, machine learning techniques in the form of a classifier are used. From this classifier's point of view, a user goes along with some kind of data modeling his or her interest, which is called the user profile in this context. After training, the classifier is supposed to receive one user profile and one article as an input, the output should be the statement if that article is interesting for the specific user or if not.

By having a user profile, the classifier should be able to recommend each user individual articles rather than the general recommendation of (for example popular) articles to unknown users.


### The data collection process

We encountered the problem that there is no pre-existing user data that is tailored to our research problem; the data would be required to contain user's ratings of news articles and would need to be restricted only to articles that are listed in Wikinews via the KnowledgeStore database.
In order to resolve this problem, we have chosen to auto-generate a set of artificial user profiles. Each artificial user is interested in a certain topic that can be rather narrow (e.g. "Pope John Paul II") or rather broad (e.g. "Germany"). However, the topic is not explicitly named in the profile, but the profile would consist of a list of articles on that topic. The goal is then to recommend articles that are about that specific topic. Creating over-simplified profiles has the advantage that one can more easily tell whether the output of the recommender system is appropriate. On the other hand, we are not able to tell whether our model would be able to cope with the complexity of real users‘ preference patterns. Therefore the dataset does not satisfy the desiderata of being representative.
Despite this disadvantage, it appeared to us as the most sensible way of acquiring a dataset. Alternative options would have been 1. creating user profiles manually, which would not yield enough data to train a machine learning model, or 2. using a pre-existing dataset with articles that are not listed in the KnowledgeStore database and using collaborative filtering.


### User profiles 

By choosing the automatic generation of idealized user profiles to build the data set, the way to model a user's profile is also narrowed down. As described in the preceding part of the documentation, a user profile from the point of view of the classifier is just a number of articles that match the user's interests. When applied to a real-world scenario, these articles could, for instance, be articles which the user read untill the end or articles that the user specified that they liked them.

### Automatic generation of the dataset

However, in order to acquire the data set, an intermediate step to model the interests of users is taken. Interests are defined as a small number of topics or categories. To enable an automatic generation of the dataset, we used the news categories provided by Wikinews as possible topics of interest. As the amount of top-level categories is low (namely, 16), we use the much bigger number of subcategories. The top-level categories are explicitly not included in the list of possible interests of users, because the thematic range of articles belonging to that category would be high compared to articles belonging to a subcategory.

The python code for generating the dataset can be found in the file *dataset_generation.py*. It holds the method *generate_dataset(amount_users, subcategories_per_user, profile_articles_per_subcategory, liked_articles_per_subcategory, disliked_articles)* which creates a desired number of user profiles. For the user profile, the number of categories of interests and the number of articles for each interest are parameters of the named method, followed by article amounts to create the data for training, validation and test. As described in the chapter [Balanced vs. imbalanced data](#balanced-vs-imbalanced-data), the dataset is imbalanced in favor of a bigger amount of uninteresting articles in comparison to interesting articles. The named parameters control this ratio and can be changed for creating a balanced dataset.
Having the same amount of profile articles as well as positive and negative samples ensures a uniform format of all users. The categories of interest themselves are not part of the dataset. 
The user's categories of interests are drawn from a weighted random distribution where categories that contain a larger number of articles are more likely to be drawn than categories containing a smaller number of articles. This decision was made because we argued that, in general, a category that contains many articles is more important and more people are interested in that topic. Subsequently, for each of the user's topics of interest, a specified number of articles from that category are randomly drawn.
The dataset is returned as a nesting of lists holding URIs. It is then saved as a pickle-file, as an easy way of serializing the structure and content of the dataset. For a more general usage of the dataset, it would of course be possible to save it as for example a csv-file. 

The dataset is structured as follows (example for two users). Each line is one step down the hierarchy of lists, so going down shows the unpacked version of the line before.

<pre>
[                                                       dataset                                  ]

[ [                         user1                                 ] , [         user2          ] ]

[ [ [   profile    ] , [               training                 ] ] , [ [profile] , [training] ] ]

[ [ [liked articles] , [ [liked articles] , [disliked articles] ] ] ,           ...              ]
</pre>

An example of what one sample of the dataset looks like can be found [here](figures/example_user.txt)

### Obtaining and storing subcategories

The python code for generating users uses the method *create_category_articles_dictionary()*, located in *knowledgestore/ks.py* as an extension of the functionality of *ks.py*. *create_category_articles_dictionary()* creates a dictionary that matches news articles to the subcategories to which they belong. The keys of that dictionary are the categories, it's values are lists of the articles that fall into that topic.
In order to obtain the subcategories for a news article, the method *get_all_news_subcategories(resource_uri)* looks at the HTML code of the corresponding Wikinews article website, given by the parameter *resource_uri*. The HTML code of such an article contains the string *wgCategories*, which is followed by a list of categories that article belongs to. 
The extraction of categories for all articles took roughly two hours, so the resulting dictionary is stored in the file *subcategory_resource_mappings.pickle*. If *create_category_articles_dictionary()* finds that file, it loads the dictionary from there instead of creating it anew.


### Excluded categories

We chose to exclude some categories, which are not taken into account for the generation of the dataset. The reason is mostly the assumption that the news articles in these categories don't share much thematic similarity. These categories are:

* the 16 top-level categories ("Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business", 
                                "Education", "Environment", "Health", "Local only", "Media", "Obituaries", 
                                "Politics and conflicts", "Science and technology", "Sports", "Wackynews", "Weather", "Women")
* all categories that contain less than 21 (*subcategories_per_user \* profile_articles_per_subcategory + subcategories_per_user \* liked_articles_per_subcategory*) because they don't hold enough articles to create a user as described above
or less than twice as many as the variable that denotes the number of articles that are chosen in the user profile per topic.
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


### Specification of parameters
The values of the parameters introduced above are fixed as follows. Note that they are mostly chosen arbitrarily.

The dataset consists of 980 user profiles. Each user is interested in three of these subcategories, which provide five news articles each to define a user profile. That gives a total of 15 articles to describe the interest of a user.
Apart from that, there are also 6 positive and 192 negative classification example articles per user which can later be used for training, validation and testing of the classifier.


### Splitting up the dataset

As the dataset is auto-generated, it is big (theoretically limited only by the size of Wikinews/amount of article there), so the use of cross-validation doesn't seem necessary for creating our classifier. The same argument counts against the usage of the same data for training, and validation test. As there is a lot of data present in the set, we use different parts of the set for training, validation and test.
Nevertheless, the division of the dataset needs additional consideration. In class, we discussed the example of splitting up the dataset for summarization: there should be some articles that are not known for the classifier during training. We do same for our classifier for news recommendation.
*dataset_splitting.py* splits the dataset provided by *dataset_generation.py* into data for training, validation and test. In order to provide some unknown data for validation and testing, a certain number of news articles is randomly chosen from all articles present in the dataset. The numbers are defined by the parameters *n_validation_articles* and *n_test_articles* of the method *split_dataset()*. Users that have one of those articles in their profile data or their positive or negative samples are selected for validation or testing data. Users that include articles chosen for validation as well as articles chosen for testing are discarded, the rest of them are selected for training data.
The splitted dataset is saved in the file *splitted_dataset.pickle*.


### Balanced vs. imbalanced data

In a real-world scenario, the amount of articles that a specific user finds uninteresting would be much higher than the number of articles that they like. For this reason, it would be problematic to have a balanced dataset, where each user has as many positive examples as they have negative examples. Therefore we created an imbalanced dataset where the percentage of articles the user is theoretically interested in is not 50% but 3.1%. We calculated the number of 3.1% by counting the average amount of articles that belong to the user's 3 categories of interest divided by the number of all articles.
Disregarding the character of automation for generating the dataset, this ratio is the closest model we have for a real-world scenario. Following that argumentation, the data for validation and testing of the classifier should be the original dataset. That means that the probability of the classifier classifying an article as negative (uninteresting for the specific user) is supposed to be much higher than the other case.
However, with the ratio of 3.1% in mind, the choice of data for validation and testing is more difficult. There are a number of downsides that go along with an imbalanced dataset like the one described above. For example, the classifier could learn to always classify an article as negative as that is the strong bias of the training data.  
However, also the use of a balanced dataset that does not model the supposed real-world scenario is not ideal. It could result in an unrealistic bias to classify more articles as interesting for the user than it would be the case in practice.
Yet, the strongest argument for choosing balanced or imbalanced to train the classifier is the actual performance of that classifier. Therefore, the results shown in the chapters [Train & Test with balanced data](#train--test-with-balanced-data) and [Train with balanced, test with imbalanced data](#train-with-balanced-test-with-imbalanced-data) state the performances of named classifiers using first a balanced and then an imbalanced dataset for training. As explained above, the data for evaluating the classifier's performance is imbalanced.


### Features

When deciding on which features to use, we decided to take word embeddings as well as term frequency - inverse document frequency (tf-idf) into account. Our first idea was to train the word embeddings over all articles of Wikinews. However, this would be very computationally expensive, wherefore we decided to use the GoogleNews word2vec (https://code.google.com/archive/p/word2vec/) word embeddings instead. This should be justifiable because the GoogleNews word2vec has also been trained on news articles. 

The sum of the embeddings of the words (or the important words according to tf-idf) can be used as a feature. As stated in the seminar, the sum of the word embeddings of a document retrieves an "average meaning" of the document, wherefore we think that it might be a meaningful feature. News articles with similar meaning should accordingly show embedding vectors that have a small cosine distance to each other.
Another idea to extract features to use for the classifier is to compute the tf-idf scores for all words in the article to be classified and use the 5 (or 10) words with the highest tf-idf value. When aiming to figure out if a new article/* is interesting for the user, the tf-idf values for those words in the new article are computed and summed up. This value can also be used as feature for the classifier. We chose to use these words with high tf-idf values because we think that the overall topic of an article can be summarized by the "most important" words of the specific article. An article which should be classified positive would yield a high sum of tf-idf values for the words that have been found earlier in the user profile, whereas the sum would be small for an uninteresting article.

In order to extract the feature vectors, we first calculated a number of measures for each article of the user's profile and for the article that should be classified - for training, this article is denoted in the dataset as a positive or negative examples for the corresponding user. 

The vector consists of the following measures:  

1\. A weighted sum of the word embedding vectors of all words in the article - they are weighted according to their tf-idf scores  
2\. An unweighted sum of the word embedding vectors of all words in the article  
3\. The sum of the word embedding vectors of the five words with the highest tf-idf scores   
4\. The weighted sum of the word embedding vectors of the five words with the highest tf-idf scores with tf-idf scores as weights  
5\. and 6\. The same as 3 and 4, but with ten words  
Apart from computing the GoogleNews word2vec word vectors, we did the following:  
7\. Selecting the five words with the highest tf-idf score from the new article and calculating the sum of the tf-idf scores of these words in the profile articles (one sum for each profile article)  
8\. Counting the number of characters in the articles and calculating the differences in length between each profile article and the new article  

For all of the vectors of 1-6, we calculate four cosine similarities: 

a. The minimum cosine similarity to the vector of the new article to the vectors articles in the profile  
b. The maximum cosine similarity to the vector of the new article to the vectors articles in the profile  
c. The mean cosine similarity to the vector of the new article to the vectors articles in the profile  
d. The average of the three highest cosine similarities

Respectively, for the values of 7 and 8 we also calculate a. the minimum, b. the maximum., c. the mean and d. the mean of the three highest values

\* when using the term "new article", we mean an article that is not included in the user's profile, but in the user's training examples. We don't mean that it is a recently published article, nor that it has not occurred before in other user's profiles during training.

## Feature selection / Dimensionality reduction
![Feature scores](https://github.com/lbechberger/ML4NLP/blob/alpha/figures/Feature_importances.png)

Combining all features results in a 32 dimensional vector. However, it is likely that not all features are equally important for the classifier to correctly classify a news article given a specific user. In order to estimate which features help the classifier the most, we apply filter and embedded methods to the extracted features.

*feature_selection.py* uses functions from the python library *sklearn* for the feature selection. The filter method that is used is named *sklearn.feature_selection.SelectKBest*, the score function is *sklearn.feature_selection.mutual_info_classif*. The latter function computes the mutual information between the features and the class (interesting or not interesting for the user). The resulting values can be used as a measure of the feature importance. The function *SelectKBest* then returns the features with the highest mutual information score.
Apart from that, a random forest classifier (*sklearn.ensemble.RandomForestClassifier*) is used for feature selection as an embedded method. Sklearns's implementation of the random forest classifier has a built-in function called *feature_importances_* that returns the feature importances. With these values, one can select the features that are most important according to the random forest classifier.

The figure above shows the sorted scores of all 32 features. As the two feature selection methods result in two differently ordered ratings of the importance of the features, the two rankings are represented independently in the figure. The scores according to the filter method are stated by blue dots, the red ones show the scores calculated by the embedded method. Note that the red dots conceal some of the blue dots especially in the left half of the figure. Also note that features with the same value on the horizontal axis aren't necessarily the same features. The sorting is done for both methods separately, as to be able to separately decide how many features to retain for each method.
In order to decide which features to use, one can look define a threshold for the importance score. It is difficult to interpret the exact values of the scores, but their distribution for one selection method can help to set such a threshold. For the embedded selection method (red dots), one clear jump in the score distribution occurs before the fifth important feature (between 26 and 27 on the horizontal axis). For the filter method selection (blue dots), the decision how many features to use is more randomly, but is based on the jump in the score distribution between feature 16 and 17. In total, five features selected by the filter method and 15 features selected by the embedded method are used. Interestingly, all five features from the first method are included in the 15 features selected by the second method.

The five features selected by both methods are the following:

* The cosine similarity between the unweighted summed word embedding of the article to be classified and the one of the most similar article of the user profile

* The maximum cosine similarity between the sums of the unweighted word embedding vectors of the ten words with the highest tf-idf scores of the profile articles and the one of the article to be classified.

* The maximum cosine similarity between the sums of the word embedding vectors of the ten words with the highest tf-idf scores of the profile articles and the one of the article to be classified. The word embeddings of the ten words are weighted by their tf-idf-score when summed.

* The maximum sum of tf-idf scores, after searching each profile article for the five words from the article to be classified that have the highest tf-idf score and summing up the tf-idf scores of these five words for each profile article separately.

* The mean of three highest similarities as explained as follows: The cosine similarity between the sums of the word embedding vectors of the ten words with the highest tf-idf scores of the profile articles and the one of the article to be classified. The word embeddings of the ten words are weighted by their tf-idf-score when summed.


### Interpretation of selection results

As one can see, the filter and embedded feature selection methods tend to recognize maximum cosine similarities as important features, rather than minimum or mean cosine similarities of all similarity scores. One possible explanation for this observation is that each user profile consists of three randomly chosen topics which are mostly not strongly related to each other. Another article as a sample to be classified as positive belongs to one of those three topics, so the similarity between the summed word embeddings of that article and each article from the same topic in the user profile is relatively high. The similarity to the news articles belonging to other topics is probably low as is the maximum similarity for a sample classified as negative.

Some insights can be gained by examining the nearest words of summed word embeddings of articles. The gensim python library provides a method *most_similar()*, which takes a word2vec embedding as an input and gives a list of words with vectors that are mathematically close to that input. The summed vector over all words of an article doesn't exactly match an vector of an existing word, but the environment of the summed embedding gives a feeling of what the embedding expresses.
The following table shows the five closest words to different summed word vectors for the news article "Government of the Bahamas isssues warning over Hurricane Hanna
" (http://en.wikinews.org/wiki/Government_of_the_Bahamas_isssues_warning_over_Hurricane_Hanna). "Weighted" means that while summing word embeddings, each embedding is multiplied (so weighted) with it's tf-idf score. "All words" says that every known word of the article is taken for summing while "top 5 words" only uses the five words of the article with the highest tf-idf-scores.
The different ways of summing are parts of four features used for the classifier.

| weighted, all words | unweighted, all words | weighted, top 5 words | unweighted, top 5 words |
|---------------------|-----------------------|-----------------------|-------------------------|
| hurricane           | the                   | hurricane             | hurricane               |
| Hurricane           | hurricane             | southeastern          | storm                   |
| Bahamas             | By_MaltaMedia_News    | tropical_storm        | southeastern            |
| hurricanes          | that                  | storm                 | hurricanes              |
| Hurricane_Wilma     | By_Jennifer_LeClaire  | northeastern          | tropical_storm          |


### Scores of classifiers


### Additional feature of shared entities

We had the idea to implement the similarity of named entities in news articles as an additional feature. The KnowledgeStore database holds information about mentions in Wikinews articles. One type of a mention is the reference to an entity, which gives a link to a DBpedia entry. Entities can be for example persons or places. A similarity measure between articles is the share of entities that are named in the articles.
The mentions of an article can be accessed via the property 'ks:hasMention'. If a mention refers to an entity, that entity can be retrieved using the property 'ks:refersTo' of the mention. Oddly enough, when a mention has the type 'nwr:EntityMention', the property 'ks:refersTo' often doesn't lead to an entity, which lessens the amount of recognized entities in an article. Consequentially, the similarity of two articles measured by the share of entities that are named in the articles is reduced.
Nevertheless, we implemented the feature of common entities. It can be turned on via the parameter *use_entity_feature* in the feature_extraction.py script. Yet, when using that feature, the time to calculate the features rapidly rises. In our test runs, enabling the entity feature led to an increase of the time needed to extract all features by the factor of 12. Therefore, we didn't use that feature in the final version of *feature_extraction.py*.


### Classifier selection

As suggested in the seminar, we used different classifiers with their default settings to work with our selected features. The classifiers with the highest scores (Cohen's Kappa) were the random forest classifier and maximum entropy. Afterwards we ran a grid search on the random forest concerning the following parameters: *n_estimators*, *max_features*, *max_depth*, *min_samples_split*, *min_samples_leaf*, *bootstrap*, but the grid search never came to an end. So, we dropped some parameters of the grid search, namely *min_samples_split*, *min_samples_leaf*, *bootstrap* and came to a score not higher than the score using the default parameters. We also ran a grid search on the maximum entropy classifier with the parameter *solver* and on the k-nearest neighbors with the parameters *n_neighbors* and *p* (power parameter for the Minkowski metric).

We also included k-nearest neighbors, support vector machine and multi-layer perceptron classifiers in order to have a comparison.


### Evaluating the classifier's performance

For evaluating the classifier's performance, we use several metrics. As Precision and Recall aren't that meaningful for themselves, we want to use the F-score as a combination. The balanced F1-score is used, as well as the F2-score which weights recall higher than precision. The reason is that for each user the number of positive examples is much lower than the amount of negative ones. For this reason, the error of not recommending an article that would be interesting to the user is more severe than the error of recommending articles that are not interesting.
Two other metrics that we use are Matthews correlation coefficient<sup>1</sup> and Cohen's Kappa<sup>2</sup>. Being somewhat similar, both are appropriate scores for evaluating the classifiers' performance. Matthews correlation coefficient has the advantage over the F-scores that it does not matter which class is defined as positive and which as negative. Moreover, both Matthews correlation coefficient and Cohen's kappa are well-suited for imbalanced data.
At last, the accuracy should also be calculated for having a metric that is widely used and intuitive.

The metrics are calculated as follows, given the confusion matrix:

Accuracy = (tp+tn)/(tp+fp+fn+tn)

Precision = tp/(tp+fp)  
Recall = tp/(tp+fn)  

F1-score = 2\*Precision\*Recall/(Precision+Recall)  
F2-score = (1+4)\*Precision\*Recall/((4\*Precision)+Recall)  

total = (tp+fp+fn+tn)  
randomAccuracy = ((tn+fp)\*(tn+fn)+(fn+tp)\*(fp+tp))/(total\*total)  
Cohen's Kappa = (accuracy-randomAccuracy)/(1-randomAccuracy)  

Matthews correlation coefficient = (tp\*tn-fp\*fn)/(sqrt((tp+fp)\*(tp+fn)\*(tn+fp)\*(tn+fn)))

### Baselines

We are comparing the classifiers' results to the baselines *always true, always false, 50-50, label frequency* as suggested during class. The resulting metrics for these baselines are as follows with respect to our imbalanced data (3,1 % positive, 96,9 % negative examples):


|  | Always “True” | Always “False” | 50-50 | Label Frequency |
|-----------------------|---------------|----------------|-------|-----------------|
| Accuracy | 0.031 | 0.969 | 0.5 | 0.94 |
| F1-Score | 0.06 | 0 | 0.058 | 0.031 |
| F2-Score | 0.138 | 0 | 0.124 | 0.031 |
| Matthews correlation coefficient | 0 | 0 | 0 | 0 |
| Cohen's kappa | 0 | 0 | 0 | 0 |

Interestingly, the always-false-baseline yields bad results in every metric except accuracy, which is counter-intuitive because as 96,9% of the samples are negative, one would assume that the always-false-baseline yields good results.


### Results

In the following are scores of the classifiers for 980 samples with first the named one feature that has been selected as the most important by both mutual information and embedded feature selection methods, and then 5 best features according to embedding feature selection methods, the 6 best features according to both mutual information and embedding features selection methods, and the 15 best according to mutual information. The tables show the scores for the classifiers with default parameters and for the parameters that have been found through a grid search (the grid search used Cohen's kappa as metric).

number of features: 1  

| Classifier                                    | Accuracy           | F1-score           | F2-score           | Cohen's kappa      | Matthews correlation coefficient |
| --------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------------------- |
| kNN with default_parameters                   | 0.96875            | 0.7368421052631581 | 0.6511627906976744 | 0.7210775317298519 | 0.7413826388009128               |
| MaxEnt with default_parameters                | 0.9692708333333333 | 0.730593607305936  | 0.6289308176100629 | 0.715548413017276  | 0.7463811137542492               |
| RF with default_parameters                    | 0.9552083333333333 | 0.6717557251908397 | 0.6480117820324005 | 0.6478161272571378 | 0.6492112258447845               |
| SVM with default_parameters                   | 0.96875            | 0.7297297297297297 | 0.6338028169014084 | 0.7142616192833721 | 0.7411024243957158               |
| MLP with default_parameters                   | 0.96875            | 0.7321428571428572 | 0.6396255850234008 | 0.7165703038504121 | 0.741086806037908                |
| kNN with {'n_neighbors': 12, 'p': 1}          | 0.9697916666666667 | 0.7387387387387386 | 0.6416275430359937 | 0.7237862319739263 | 0.7509849567421718               |
| MaxEnt with {'solver': 'newton-cg'}           | 0.9697916666666667 | 0.7363636363636364 | 0.6357927786499216 | 0.7215178477650908 | 0.7512356762378097               |
| RF with {'max_depth': 460, 'n_estimators': 6} | 0.9515625          | 0.654275092936803  | 0.641399416909621  | 0.6282633134862805 | 0.6286702470021618               |

number of features: 5  

| Classifier                                     | Accuracy           | F1-score           | F2-score           | Cohen's kappa      | Matthews correlation coefficient |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------------------- |
| kNN with default_parameters                    | 0.9677083333333333 | 0.7280701754385964 | 0.6434108527131782 | 0.711780116120847  | 0.7318234135929149               |
| MaxEnt with default_parameters                 | 0.9703125          | 0.7532467532467532 | 0.6712962962962963 | 0.7381467373619433 | 0.7559784828140533               |
| RF with default_parameters                     | 0.9692708333333333 | 0.7400881057268723 | 0.6521739130434784 | 0.7246314806891991 | 0.7460754030840211               |
| SVM with default_parameters                    | 0.96875            | 0.7321428571428572 | 0.6396255850234008 | 0.7165703038504121 | 0.741086806037908                |
| MLP with default_parameters                    | 0.9697916666666667 | 0.75               | 0.6702619414483821 | 0.7345955298794526 | 0.7514311021575419               |
| kNN with {'n_neighbors': 4, 'p': 2}            | 0.9671875          | 0.7200000000000001 | 0.6308411214953271 | 0.7035962479048842 | 0.7265463899492495               |
| MaxEnt with {'solver': 'newton-cg'}            | 0.9703125          | 0.7532467532467532 | 0.6712962962962963 | 0.7381467373619433 | 0.7559784828140533               |
| RF with {'max_depth': 310, 'n_estimators': 11} | 0.9671875          | 0.7248908296943231 | 0.6424148606811145 | 0.7082911300824772 | 0.727256801230567                |

number of features: 6  

| Classifier                                     | Accuracy           | F1-score           | F2-score           | Cohen's kappa      | Matthews correlation coefficient |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------------------- |
| kNN with default_parameters                    | 0.9677083333333333 | 0.7256637168141594 | 0.6376360808709176 | 0.7094700464203605 | 0.7315194091497506               |
| MaxEnt with default_parameters                 | 0.9703125          | 0.7532467532467532 | 0.6712962962962963 | 0.7381467373619433 | 0.7559784828140533               |
| RF with default_parameters                     | 0.9666666666666667 | 0.7241379310344828 | 0.6471494607087827 | 0.707139895039396  | 0.7233462348950305               |
| SVM with default_parameters                    | 0.96875            | 0.7321428571428572 | 0.6396255850234008 | 0.7165703038504121 | 0.741086806037908                |
| MLP with default_parameters                    | 0.9697916666666667 | 0.75               | 0.6702619414483821 | 0.7345955298794526 | 0.7514311021575419               |
| kNN with {'n_neighbors': 19, 'p': 1.5}         | 0.9708333333333333 | 0.7477477477477478 | 0.649452269170579  | 0.7333108446644806 | 0.7608674890886279               |
| MaxEnt with {'solver': 'newton-cg'}            | 0.9703125          | 0.7532467532467532 | 0.6712962962962963 | 0.7381467373619433 | 0.7559784828140533               |
| RF with {'max_depth': 260, 'n_estimators': 71} | 0.9666666666666667 | 0.7217391304347827 | 0.6414219474497681 | 0.7048296669244923 | 0.7227627614926289               |

number of features: 15  

| Classifier                                     | Accuracy           | F1-score           | F2-score           | Cohen's kappa      | Matthews correlation coefficient |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------------------------- |
| kNN with default_parameters                    | 0.9713541666666666 | 0.7533632286995516 | 0.65625            | 0.739135590205727  | 0.7656457289343342               |
| MaxEnt with default_parameters                 | 0.9713541666666666 | 0.759825327510917  | 0.6733746130030959 | 0.7453335262624801 | 0.7652910690783761               |
| RF with default_parameters                     | 0.9697916666666667 | 0.7456140350877193 | 0.6589147286821706 | 0.7303749473388569 | 0.7509418640089106               |
| SVM with default_parameters                    | 0.9692708333333333 | 0.7354260089686099 | 0.6406249999999999 | 0.72016363312978   | 0.7459933159032132               |
| MLP with default_parameters                    | 0.9697916666666667 | 0.7363636363636364 | 0.6357927786499216 | 0.7215178477650908 | 0.7512356762378097               |
| kNN with {'n_neighbors': 2, 'p': 1}            | 0.9697916666666667 | 0.7387387387387386 | 0.6416275430359937 | 0.7237862319739263 | 0.7509849567421718               |
| MaxEnt with {'solver': 'newton-cg'}            | 0.9713541666666666 | 0.759825327510917  | 0.6733746130030959 | 0.7453335262624801 | 0.7652910690783761               |
| RF with {'max_depth': 135, 'n_estimators': 51} | 0.9723958333333333 | 0.7705627705627704 | 0.6867283950617284 | 0.756522404915491  | 0.7747980597012636               |

One can see in the table that the best performance in all metrics was achieved by the random forest classifier that has been optimized through a grid search while using 15 parameters. The second best is the maximum entropy classifier. These classifiers clearly beat all of the baselines. One can also see that for some classifiers it does not make a big difference how many features are used (k-nearest-neighbors), whilst for others (e.g. the random forest) it makes a big difference.


### Missing data
Luckily, the features we are using are not prone to produce missing data. On the one hand, this is due to the fact that the feature extraction is independently of the amount of articles in the user profile (as long as there is at least one article), on the other hand, the feature extraction only takes the raw text of articles, so missing additional information (like DBpedia-information) is not an issue.
Nevertheless, one potential problem is the lack of word2vec-embeddings for rare or special words. In particular, when the embeddings of the five words with the highest tf-idf scores are calculated and summed up as a feature, it can happen there isn't a word2vec-embedding for any of the words. With possible high tf-idf scores for generally rare words, the probability for not having word2vec-embeddings for any of those five words is even elevated. Missing embeddings are replaced with a null-vector (model["for"] * 0), as to avoid missing values for features. Even so, a null-vector weakens the informative value of the corresponding feature and might lead to falsification of the feature itself.

## Train & Test with balanced data

As mentioned in chapter [Balanced vs. imbalanced data](#balanced-vs-imbalanced-data), we also tried to train and test the classifiers with a dataset that contains a balanced amount of positive and negative examples per user. These are the results (performance is measured in Cohen's kappa):


* kNN with default parameters: 0.532319391634981 
* kNN with {'n_neighbors': 12, 'p': 1}:  0.5871559633027523   
* MaxEnt with default parameters: 0.7518796992481203 
* MaxEnt with {'solver': 'newton-cg'}: 0.7515060240963856  
* RF with default parameters: 0.651685393258427  
* RF with {'max_depth': 210, 'n_estimators': 21}:  0.6970830216903516  
* SVM with default parameters: 0.7174076865109269  
* MLP with default parameters: 0.5896656534954408  

One can see that in most cases, the performance is lower than in the imbalanced setting. However, the maximum entropy classifier reaches a very large performance on the balanced data.


## Train with balanced, test with imbalanced data

These are the results of training on balanced data and testing on imbalanced data (again, performance is measured in Cohen's kappa):

* kNN with default parameters: 0.25806660409722004  
* kNN with {'n_neighbors': 7, 'p': 1}: 0.2621448212648946  
* MaxEnt with default parameters: 0.5747151266403845  
* MaxEnt with {'solver': 'newton-cg'}: 0.5807470325601656  
* RF with default parameters: 0.4649720475192174  
* RF with {'max_depth': 410, 'n_estimators': 41}: 0.5351925630810093  
* SVM with default parameters: 0.6124581227641814  
* MLP with default parameters: 0.5914396887159533  

One can see that these results are not as good as those that were reached by training with imbalanced data.


### Setup (How to use our code)

The code is written in Python 3 (https://www.python.org/ ). In order to run the code, one must set up a Python environment with the following packages installed:

* numpy (www.numpy.org/ )  
* sklearn (https://scikit-learn.org/)  
* matplotlib (https://matplotlib.org/)  
* gensim (https://radimrehurek.com/gensim/ )  
* nltk (https://www.nltk.org/)  

(Note that the modified version of KnowledgeStore (*ks.py*) that is in the alpha branch of the repository is required.)

The first python program to be run is *dataset_generation.py*. It saves the created dataset as *dataset.pickle*. Afterwards, the program *dataset_splitting.py* splits the dataset into training, validation and test data and saves it as *splitted_dataset.pickle*. The program *feature_extraction.py* uses this splitted dataset to compute the features and saves them as *featurised_dataset.pickle*. After this step, feature selection is applied by the program *feature_selection.py*, and the resulting datasets are saved as *reduced_datasetN.pickle* (N stands for the number of features that have been selected). The program *classifiers.py* applies the different classifiers to the feature-selected dataset and saves the performance of the different classifiers into the *classifier_resultsN.pickle* files. Eventually, the script results.py prints them into a table.

### References

<sup>1</sup> Matthews, B. W. (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme". Biochimica et Biophysica Acta (BBA) - Protein Structure.  
<sup>2</sup> Cohen, J. (1960): A coefficient of agreement for nominal scales. In: Educational and
psychological measurement, Bd. 20(1): S. 37–46.
