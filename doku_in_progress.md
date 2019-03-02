
# ML4NLP - Alpha
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Alpha.


Project goal
 User profile
Data collection
 Consideration of options

 Automatic generation of the dataset
  Excluded categories
 Modeling of user profile
 Structure of dataset

Splitting of dataset

Evaluating metrics
Baselines

Balanced/Imbalanced

Features

Dimensionality reduction

Performances of different classifiers
Comparison to baselines ??

Missing data

Classifier selection
Hyperparameters, Gridsearch
(Balanced/Imbalanced)

Training, Test, Validation ??

Final result

How to setup environment, structure of files, etc.
dataset_generation.py
link to sourcecode

### Project goal

Goal of the project is the recommendation of news articles on the basis of Wikinews (https://en.wikinews.org/wiki/Main_Page), accessed via KnowledgeStore API (http://knowledgestore2.fbk.eu/nwr/wikinews/ui). The latter includes the enrichment of articles with information in a format that makes them easiy to process by computers, for example the linking of mentioned entities with DBpedia-entries (https://wiki.dbpedia.org/).

One main approach to solve the task of news recommendation is based on collaborative filtering, another one is based on the matching of a user's profile to an article. The former approach relies on a database of preferrably many users and their interests in articles, the latter doesn't require the knowledge about other users after training. The decision, which of these approaches to use for the project, is taken for the matching of a user's profile to an article. Neither Wikinews nor the additional information from the KnowledgeStore database include user data, additionally, using another dataset like the yahoo-news-dataset (foto von christopher??) would be beyond the seminar's scope. Nevertheless, for training, some kind of user data is required. This topic will be revisited in (???)

In order to estimate a user's interest in a given article, machine learning techniques in the form of a classifier are used. From this classifier's point of view, a user goes along with some kind of data modeling his or her interest, which is called the user profile in this context. After training, the classifier is supposed to receive one user profile and one article as an input, the output should be the statement if that article is interesting for the specific user or if not.

By having a user profile, the classifier should be able to recommend each user individual articles rather than the general recommendation of (for example popular) articles to unknown users.



## Session 3, 06.11.18

### The data collection process

We encountered the problem that there is no pre-existing user data that is tailored to our research problem (??genauer, no preexisting); the data would be required to be restricted only to articles that are listed in Wikinews via the KnowledgeStore database.
In order to resolve this problem, we have chosen to auto-generate a set of artificial user profiles. Each artificial user is interersted in a certain topic that can be rather narrow (?? example) or rather broad (?? example).  However, the topic is not explicitely named in the profile, but the profile would consist of a list of articles on that topic. The goal is then to recommend articles that are about that specific topic. Creating over-simplified profiles has the advantage that one can more easily tell whether the output of the recommender system is appropriate. On the other hand, we are not able to tell whether our model would be able to cope with the complexity of real users‘ preference patterns. Therefore the dataset would not satisfy the desiderata of being representative.

(?? design decisions begründen)
When applied to a real-world application, a user profile can be seen as a set of articles that the user either read till the end or articles that the user specified that they liked them.





### User profiles (Einmal Resultat statt schrittweis??)

By chosing the automatic generation of idealized user profiles to build the data set, the way to model a user's profile is also narrowed down. As described in the preceding part of the documentation, a user profile from the point of view of the classifier is just a number of articels that match the user's interests. To give an example, these articles could be articles which the user read til the end.

### Automatic generation of the dataset

However, in order to aquire the data set, an intermediate step to model user's interests is taken. Interests are defined by a small number of topics the user is interested in. To enable the automatic generation, the topics of interest match the categories into which the articles of Wikinews are sorted. As the amount of top level categories is low (namely, 16), we use the much bigger number of subcategories. A user profile consists of 15 articles that belong into the topics of interest for a user. In order to generate the user profiles, we created a python dictionary that lists for each topic all articles that fall into that topic. The articles are drawn from a random distribution where each topic is weighted by the number of articles that belong to that topic. We chose this approach because we assume that topics that contain a large number of articles are more important as more people are interested in these topics.

(?? noch sagen, welche Kategorien-Größen ausgeklammert werden)
(?? an example vector would be good)




Folgendes als einzelnen Teil, Codenahe Doku??)
 *dataset_generation.py* holds the method *generate_dataset(amount_users, subcategories_per_user, articles_per_user_and_category)* which creates a desired number of user profiles, each one consisting of a defined number of categories of interests per user and a defined number of articles per user's interest. This ensures a uniform format in which each user profile is defined by an equal amount of articles the user liked (the user's categories of interest are not specified in the dataset). The user's categories of interests are drawn from a weighted random distribution where categories that contain a larger number of articles are more likely to be drawn than categories containing a smaller number of articles. This decision was made because we argued that, in general, a category that contains many articles is more important and more people are interested in that topic. Subsequently, for each of the user's topics of interest, a specified number of articles from that category are randomly drawn.
The dataset is returned as a list of URIs and can be saved e.g. in a csv-file.

Topics of interest are chosen from the category mapping of Wikinews. Note that the top-level categories aren't taken into account but only the level-2-categories, so subcategories. We excluded the first ones because if such a category was chosen for a user's topic of interest, the thematic range of articles belongig to that category would be high compared to articles belonging to a subcategory.

The python code for generating users uses the method *create_category_articles_dictionary()*, for which we extended *ks.py* in the *knowledgestore* folder. *create_category_articles_dictionary()* creates a dictionary that matches news articles to the subcategories to which they belong. In order to get the subcategories for an article, the method *get_all_news_subcategories(resource_uri)* looks at the HTML code of the corresponding Wikinews article website, given by the parameter *resource_uri*. The HTML code contains the string *wgCategories*, which is followed by a list of categories that article belongs to. The top-level categories are excluded as explained above. 
The extraction of categories for every article took roughly two hours, so the resulting dictionary is stored in the file *subcategory_resource_mappings.pickle*. If *create_category_articles_dictionary()* finds that file, it loads the dictionary from there instead of creating it anew.



The code for generating the dataset can be found in the file *generate_dataset.py*. 
The dataset consists of 1000 user profiles. (Imbalanced ??) Each user profile contains 30 articles the user liked - 10 from each of the 3 topics the user is interested in.
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
(ebenfalls auf imbalanced ??)
Then, for each topic of interest, there were 20 articles drawn - 10 for the user profile and 10 for the positive training samples. Afterward, 30 articles that do not belong to any of the three categories of 
interest were drawn in order to be used as negative training samples.
Finally, the dataset was saved as a pickle file.


## Session 7, 04.12.18

### Splitting up the dataset

As the dataset is auto-generated, it is big (theoretically limited only by the size of Wikinews/amount of article there), so the use of cross-validation doesn't seem necessary for creating our classifier. The same argument counts against the usage of the same data for training, test and validation. As there is a lot of data present in the set, we can use different parts of the set for training, test and validation.
Nevertheless, the division of the dataset needs additional consideration. In class, we discussed the example of splitting up the dataset for summarization: there should be some articles that are not known for the classifier during training. We want the same for our classifier for news recommendation. One solution that comes into mind is the reservation of articles that a newer than a certain date and putting them aside for test and validation data.

### Evaluating the classifier's performance

(why which scores??)
For evaluating the classifier's performance, we want to use several metrics. As Precision and Recall aren't that meaningful for themselves, we want to use the F-score as a combination.

A decision still to be made is if we use the balanced F1-score or the F2-score to weigth recall higher than precision. The reason is that for each user the number of positive examples is much lower than the amount of negative ones. Wherefore the error of not recommending an article that would be interesting to the user is more severe than the error of recommending articles that are not interesting.
Two other metrics that we want to use are Matthews correlation coefficient and Kohen's Kappa. Being somewhat similar, both are appropriate scores for evaluating the classifiers performance.
At last, the accuracy should be calculated for having a metric that is widely used.

### Baselines

### Balanced vs. imbalanced data

(?? in dataset or theroretícally?? meaning, in documentation, it's still to decide whether to use balanced or imbalanced)
Due to the automatic generation of the dataset, the ratio of positive examples for one user in the dataset can be determined relatively accurately. Each of the users the dataset contains is interested in artilces of three categories; the amout of categories that are used to generate the dataset is (??). Additionally, the average amout of articles that a category of the dataset contain is ???. Altogether, the average amount of articles that belong to the user's three categories of interest devided by the number of all articles gives a ratio of 3.1% of positive examples in the dataset. (really, maybe we just looked up the actual amount in dataset, not calculated one??).
Disregarding the character of automation for generating the dataset, this ratio is the closest model we have for a real-world scenario. Following that argumentation, the data for testing and evaluating (Reihenfolge??) the classifier's performance should be the original dataset. That means that the probability of the classifier classifing an article as negative (so uninteresting) is supposed to be much higher than the other case (see chapter ???).
However, with the ratio of 3.1% in mind, the choice of data for validation and testing (Reihenfolge und so??) is more difficult. There are a number of downsides that go along with an imbalanced dataset like the one described above. One is the possibility that the classifier learns to always classify an article as negative as that is the strong bias of the training data. (what else??)
But also the use of a balanced dataset that doesn't model the supposed real-world scenario isn't ideal. It could result in an unrealistic bias to classify more articles as interesting for the user than it would be the case in practice. (what else??)
Yet, the strongest argument for choosing balanced or imbalanced to train the classifier is the actual performance of that classifier. Therefore, the results shown in (Kapitel??) state the performances of named classifiers using first a balanced and then an imbalanced dataset for training. As explained above, the data for evaluating the classifier's performance is imbalanced.
(data??)



The dataset that we are using has a huge amount of samples, which is due to the fact that it is computationally generated. 



As baselines we are planning to use *always true, always false, 50-50, label frequency* as suggested during class. The resulting metrics for these baselines are as follows:

|  | Always “True” | Always “False” | 50-50 | Label Frequency |
|-----------------------|---------------|----------------|-------|-----------------|
| Accuracy | 0.5 | 0.5 | 0.5 | 0.5 |
| F1-Score | 0.67 | 0 | 0.5 | 0.5 |
| F2-Score | 0.8333 | 0 | 0.5 | 0.5 |
| Matthew's correlation | 0 | 0 | 0 | 0 |
| Cohen's kappa | 0 | 0 | 0 | 0 |



## Session 7, 11.12.18



### Features

When deciding on which features to use, we decided to take word embeddings as well as term frequency - inverse document frequency (tf-idf) into account. Our first idea was to train the word embeddings over all articles of Wikinews. However, this would be very computationally expensive, wherefore we decided to use the GoogleNews word2vec word embeddings instead. This should be justifiable becauses the GoogleNews word2vec has also been trained on news articles.(was zu unseen words schreiben ??) Eventually(??), the sum of the embeddings of the words (or the important words according to tf-idf) can be used as a feature. As stated in Handouts_Session_8 (?? vielleicht echte Quelle angeben) , the sum of the word embeddings of a document retrieves an "average meaning" of the document, wherefore we think that it might be a meaningful feature. Articles with similar meaning should accordingly show embedding vectors that have a small cosine distance to each other.
Another idea to extract features to use for the classifier is to compute the tf-idf scores for all words in the article to be classified and use the 5 (or 10) words with the highest tf-idf value. When aiming to figure out if a new article is interesting for the user, the tf-idf values for those words in the new article are computed and summed up. This value can also be used as feature for the classifier. We chose to use these words with high tf-idf values because we think that the overall topic of an article can be summarized by the "most important" words of the specific article. An article which should be classified positive would yield a high sum of tf-idf values for the words that have been found earlier in the user profile, whereas the sum would be small for an uninteresting article. ?? bezieht sich das auf was anderes?

In order to extract the feature vectors, we first calculated a number of measures for each article of the user's profile and for the article that should be classified - for training, this article is denoted in the dataset as a positive or negative examples for the corresponding user. 

The vector consists of the following measures:
1. a weighted sum of the word embedding vectors of all words in the article - they are weighted according to their tf-idf scores 
2. an unweighted sum of the word embedding vectors of all words in the article
3. the sum of the five words with the highest tf-idf scores (kommentar zu unseen words ?? )  
4. the weighted sum of the word embedding vectors of the five words with the highest tf-idf scores with tf-idf scores as weights 
5 and 6. the same as 3 and 4, but with ten words 
Apart from computing the word vectors, we did the following: 
7. selecting the five words with the highest tf-idf score from the new article and calculating the sum of the tf-idf scores of these words in the profile articles 
8. counting the number of characters in the article 

For all of these vectors, we calculated four cosine distances: 

a. the minimum distance to the vector of the new article to the vectors articles in the profile  
b. the maximum distance to the vector of the new article to the vectors articles in the profile  
c. the mean distance to the vector of the new article to the vectors articles in the profile  
d. the average of the three lowest distances (except from 7., where it is the average of the three highest)  


![Feature scores](https://github.com/lbechberger/ML4NLP/blob/alpha/Feature_Scores.png)

Combining all distances to a feature vector results in a 32 dimensional vector. However, it is likely that not all features are equally important for the classifier to correctly classify an article. In order to estimate which features help the classifier the most, we applied filter methods to the extracted features, namely sklearn.feature_selection.mutual_info_classif. The figure above shows the scores of all 32 features sorted by score. According to this, 14 features have a low mutual information value wherefore we assume that they would not play a big role in the classification process.


?? Schreiben, dass wir jetzt nur fünf oder so genommen haben

?? Schreiben, dass auch embedded genommen wurde


## Session 10, 15.01.19

In the past few weeks, we extracted the features from the dataset and reduced the dimensionality of the feature vector by applying filter selection methods. In order to extract features, we first calculated a number of measures for each article of the user's profile and for the article that should be classified - for training, this article is denoted in the dataset as a positive or negative examples for the corresponding user.

The first six measures are (GoogleNews-)word2vec embedding vectors as follows:
1. computing a weighted sum of all words in the article - they are weighted according to their tf-idf scores.
2. taking all words in the article and computing the sum without weighing
3. computing the sum of the five words with the highest tf-idf scores
4. computing the weighted sum of the five words with the highest tf-idf scores with tf-idf scores as weights
5 and 6. the same as 3 and 4, but with ten words
Apart from computing the word vectors, we did the following:
7. selecting the five words with the highest tf-idf score from the new article and calculating the sum of the tf-idf scores of these words in the profile articles
8. counting the number of characters in the article

for all of these vectors, we calculated four cosine distances:
a. the minimum distance to the vector of the new article to the vectors articles in the profile
b. the maximum distance to the vector of the new article to the vectors articles in the profile
c. the mean distance to the vector of the new article to the vectors articles in the profile
d. the average of the three lowest distances (except from 7., where it is the average of the three highest)

![Feature scores](https://github.com/lbechberger/ML4NLP/blob/alpha/Feature_Scores.png)

All distances combined to a feature vector gives us 32 dimensions. However, it is likely that not all features are equally important for the classifier to correctly classify an article. In order to estimate which features help the classifier the most, we applied filter methods to the extracted features, namely sklearn.feature_selection.mutual_info_classif.
The figure above shows the scores of all 32 features sorted by score. According to this, 14 features have a low mutual information value wherefore we assume that they would not play a big role in the classification process. For the next step of the machine learning research cycle, we use the reduced feature set of the remaining 18 features with more mutual information.
This makes in total 32 features. Despite mentioned in the last documentation step, we did not replace words that are not in the googlenews-word2vec-embeddings, but ignored them.


## Session 12, 29.01.19

### Scores of classifiers (Dimensionality reduction???)

As suggested in the seminar, we used different classifiers with their default settings to work with our selected features. The classifiers with the highest scores (kohen's cappa) are then investigated closer, so we tried to narrow down the best hyperparameters for those classifiers. According to mutual_inf_classif the three most important features are the ones with index 25, 1, 21 (starting with the best). These are firstly the highest tf-idf matching, secondly the highest cosine similarity of the summed word2vec-embedding of words weighed by tf-idf score and thirdly the highest cosine similarity of word2vec-embeddings of summed and unweighed words of the ten highest words of the article according to their tf-idf scores.
When fed to the different classifiers with their default parameters, it's enough to use the one most important feature according to its mutual_inf_classif to reach the scores the classifiers reached when using all features. Moreover, some of the scores reached when using all features are even improved, but not improving the scores of the best-performing classifiers. 

### Classifier selection

We decided to use the classifiers random forest and maximum entropy because these classifier yielded the best results when ran on the dataset (without hyperparameter tuning). Afterwards we ran a grid search on the random forest concerning the following parameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, but the grid search never came to an end. So, we dropped some parameters of the grid search, namely min_samples_split, min_samples_leaf, bootstrap and came to a score not higher than the score using the default parameters. Shame on the grid search.

### Results

In the following are scores of the classifiers for 700 samples with first the named one feature and then the 10 best features according their mutual information:

* One feature:

    * default parameters:
      
      **kNN** 0.6488294314381271  
        
      **MaxEnt** 0.6488294314381271  
        
      **RF** 0.5914396887159532  
        
      **SVM** 0.6488294314381271  
        
      **MLP** 0.6488294314381271  


    * Hyperparameter tuning:

        **K nearest neighbors**  
        'n_neighbors': 2  
        'p': 1  
        Performance: 0.6488294314381271  

        **Max entropy**  
        'solver': 'newton-cg'  
        0.6488  

        **Random forest**  
        'max_depth': 260  
        'n_estimators': 6  
        Performance: 0.6731517509727627  

        **MLP**  
        'activation': 'tanh'  
        'alpha': 0.05  
        'hidden_layer_sizes': (100,)  
        'learning_rate': 'constant'  
        'solver': 'adam'  
        Performance: 0.6488294314381271  



* 10 features:

    * default parameters:
       
      **kNN** 0.6488294314381271  
       
      **MaxEnt** 0.6488294314381271  
     
      **RF** 0.6731517509727627  
       
      **SVM** 0.6488294314381271  
       
      **MLP** 0.6488294314381271  


    * Hyperparameter tuning:

      **K nearest neighbors:**  
       'n_neighbors': 2  
       'p': 1  
       Performance: 0.6488294314381271  
  
      **Max Entropy:**  
       'solver': 'newton-cg'  
       Performance: 0.6488294314381271  
  
      **Random forest:**  
       'max_depth': None  
        'max_features': 'sqrt'  
        'n_estimators': 67  
        Performance: 0.5928798026083891  


### Missing data
Luckily, the features we are using are not prone to produce missing data. On the one hand, this is due to the fact that the feature extraction is independently of the amount of articles in the user profile (apart from zero articles) (heh??), on the other hand, the feature extraction only takes the raw text of articles, so missing additional information (like dbpedia-information) is not an issue.
Nevertheless, one potential problem is the lack of word2vec-embeddings for rare or special words. In particular, when the embeddings of the five words with the highest tf-idf scores are calculated and summed up as a feature, it can happen there isn't a word2vec-embedding for any of the words. With possible high tf-idf scores for generally rare words, the probability for not having word2vec-embeddings for any of those five words is even elevated. Missing embeddings are replaced with a null-vector (model["for"] * 0), as to avoid missing values for features. Even so, a null-vector weakens the informative value of the corresponding feature and might lead to falsification of the feature itself.


## Scores of classifiers

As suggested in the seminar, we used different classifiers with their default settings to work with our selected features. The classifiers with the highest scores (kohen's cappa) are then investigated closer, so we tried to narrow down the best hyperparameters for those classifiers. According to mutual_inf_classif the three most important features are the ones with index 25, 1, 21 (starting with the best). These are firstly the highest tf-idf matching, secondly the highest cosine similarity of the summed word2vec-embedding of words weighed by tf-idf score and thirdly the highest cosine similarity of word2vec-embeddings of summed and unweighed words of the ten highest words of the article according to their tf-idf scores.
When fed to the different classifiers with their default parameters, it's enough to use the one most important feature according to its mutual_inf_classif to reach the scores the classifiers reached when using all features. Moreover, some of the scores reached when using all features are even improved, but not improving the scores of the best-performing classifiers. 

## Classifier selection

We decided to use the classifiers random forest and maximum entropy because these classifier yielded the best results when ran on the dataset (without hyperparameter tuning). Afterwards we ran a grid search on the random forest concerning the following parameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, but the grid search never came to an end. So, we dropped some parameters of the grid search, namely min_samples_split, min_samples_leaf, bootstrap and came to a score not higher than the score using the default parameters. Shame on the grid search.

## Results

In the following are scores of the classifiers for 700 samples with first the named one feature and then the 10 best features according their mutual information:

One feature:

default parameters:
kNN 0.6488294314381271
MaxEnt 0.6488294314381271
RF 0.5914396887159532
SVM 0.6488294314381271
MLP 0.6488294314381271


Hyperparameter tuning:

K nearest neighbors
'n_neighbors': 2
    'p': 1
Performance: 0.6488294314381271

Max entropy
'solver': 'newton-cg'
0.6488

Random forest
'max_depth': 260
'n_estimators': 6
Performance: 0.6731517509727627

MLP:
'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'solver': 'adam'
Performance: 0.6488294314381271



10 features:

kNN 0.6488294314381271
MaxEnt 0.6488294314381271
RF 0.6731517509727627  !!!
SVM 0.6488294314381271
MLP 0.6488294314381271


Hyperparameter tuning:

K nearest neighbors:
'n_neighbors': 2, 'p': 1
Performance: 0.6488294314381271

Max Entropy:
'solver': 'newton-cg'
Performance: 0.6488294314381271

Random forest:
'max_depth': None
'max_features': 'sqrt'
'n_estimators': 67
Performance: 0.5928798026083891


## Missing data
Luckily, the features we are using are not prone to produce missing data. On the one hand, this is due to the fact that the feature extraction is independently of the amount of articles in the user profile (apart from zero articles), on the other hand, the feature extraction only takes the raw text of articles, so missing additional information (like dbpedia-information) is not an issue. Nevertheless, one potential problem is the lack of word2vec-embeddings for rare or special words. In particular, when the embeddings of the five words with the highest tf-idf scores are calculated and summed up as a feature, it can happen there isn't a word2vec-embedding for any of the words. With possible high tf-idf scores for generally rare words, the probability for not having word2vec-embeddings for any of those five words is even elevated. Missing embeddings are replaced with a null-vector (model["for"] * 0), as to avoid missing values for features. Even so, a null-vector falsifies the statement of the corresponding feature.



## Train & Test with balanced data

TRAIN & EVALUATE
kNN 0.532319391634981 {'n_neighbors': 12, 'p': 1  0.5871559633027523}
MaxEnt 0.7518796992481203 {default and best params {'solver': 'newton-cg'} (Performance: 0.7515060240963856)}
RF 0.651685393258427 {'max_depth': 210, 'n_estimators': 21  0.6970830216903516}
SVM 0.7174076865109269
MLP 0.5896656534954408




## Train with balanced, test with imbalanced data
TRAIN & EVALUATE
kNN 0.25806660409722004
MaxEnt 0.5747151266403845
RF 0.4649720475192174
SVM 0.6124581227641814
MLP 0.5914396887159533

GRID SEARCH
Best params: {'n_neighbors': 7, 'p': 1}
Performance: 0.2621448212648946

Best params: {'solver': 'newton-cg'}
Performance: 0.5807470325601656

Best params: {'max_depth': 410, 'n_estimators': 41}
Performance: 0.5351925630810093



Gleiche Schreibweise für Ausdrücke: dbpedia, knowledgestore, dataset
Sichtweise/Zeitform für am Ende vom Projekt

