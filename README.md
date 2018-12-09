# ML4NLP - Epsilon
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Epsilon.

## Week 29.10. - 04.11.

### Goal Statement
We aim to create a system that is capable of answering questions concerning a single document, e.g. when asked "Who owns the Company BigCompany?" after reading a text about this company, it should be able to answer "Mrs Bigshot". For this we make use of the  KnowledgeStore Database (https://knowledgestore.fbk.eu/), in particular the NewsReader project (http://knowledgestore2.fbk.eu/nwr/wikinews/ui) and the articles contained within.

### Approach
We quickly realised that a complete QA system capable of parsing the whole range of complex questions ("Whose fault is the decline of the Euro, according to Mrs Moneybags?") would be beyond our scope. Inspired by the triplets that make up entries in RDF (x y z), we decided to limit the questions to the style of "Who has the relation y to z?", "What is the relation between x and z?" and "To whom does x have the relation y?".
This enables us to make use of these triplets to autogenerate a large amount of questions, making heavy duty machine learning feasible.
Furthermore this decision enabled the splitting of the task into 3 variants, one for each style of question. This is especially well suited for our approach, since we consist of three groups that will be working in parallel.
Our first challenge will be the generation of a Question-Answer-Pair (QAP) database for each of the articles in the NewsReader project. The next step will be the selection of a suitable model for the prediction of correct answers to given questions on a given article. This model will be trained on the NewsReader articles with the QAP database as ground truth.

### Choice of Input and Target
We decided that the target of the predictions ought to be the completed triple of the (Subject, Relation, Object) form.
Concerning the input, our decision has not yet been finalised. While we are sure that the raw text of the article in question ought to be part of it, there are further open questions. Should parts of the KnowledgeStore database be made available to the algorithm in some form in order to represent world knowledge? Can we input the complete article, or should we use only the sentences containing two of the three words in the given triple? Should we engineer additional features using traditional NLP methods or is the raw text sufficient? Continued discussion is required here.

### Choice of Model
So far, possible models have only briefly been discussed. One possible choice would be the use of a classification algorithm which predicts for each word or semantic unit in the text whether or not it matches the missing part of the triple. This solves the overall problem of answering questions on a given article not yet completely, but provides a semantic basis for the answering of the question. When training with the QAP database, the output of the classification algorithm will have to be translated into the denominator of the corresponding node in the database. When working with a natural language question as input, the completed triple should ideally be further translated into a natural language answer by means of natural language generation.

## Week 05.11 - 11.11

### Data Collection
As discussed before, we want to make use of the data contained within the KnowledgeStore database. We desire a large amount of data to enable more complicated machine learning approaches, and we want data clean of formating/labeling issues that could break our pipeline. As such, we decided to autogenerate the data from the DB if possible. Furthermore, we decided that we want to focus on the Relation component of the triples, i.e. the answer the question "What is the relation between Mr Bigshot and the BigCompany." -> "Mr Bigshot owns the BigCompany/The BigCompany is owned by Mr Bigshot."
Researching the database and the entries in the database revealed that there is one kind of entry in the knowledgestore database, which is called "RelationMention" and seems to have nearly ideal properties for the generation of QA pairs (see https://knowledgestore.fbk.eu/ontologies/newsreader.html#RelationMention). However, the given RESTful API seems to be unable to determine the rdf:type of a mention, which makes it difficult to filter for RelationMentions. Currently we are exploring a combination of mention, resource and SPARQL queries to nevertheless get a list of RelationMentions, their article and their attributes.
From these filtered queries we desire to generate the test and trainings data sets. However, as mention in "Choice of Input and Target" we are not quite sure if we also want to add real world knowledge about the entries in the article via further SPARQL queries to dbpedia.

###  Pros and Cons
As mentioned, we wanted a large and syntactically clean data set, which is possible with autogeneration. However, we are aware of the limitations. We cannot surpass the semantic quality of the KnowledgeStore, nor can we generate information not yet contained within it, possibly an issue if we want to apply the algorythm to articles outside of the DB. For the articles contained within, however, we are quite sure to be capable of generating representative QA pairs, as these articles are related in their content and style.
Objects to label: Words (possibly mentions) which complete the given incomplete triple. NWR:RelationMention usually has the required labeling information, but as mentioned before, we are unable to access it for now.

### Properties of the Data Set

Is it large?
The size is determined by the size of the NewsReader data set, which currently contains 19751 articles with each on average 263 mentions per article. How many of these are RelationMentions, we are not able to say at this point, since we cannot filter for them.

Is it representative of task and data distribution?
Representativeness for our task is given, since we select information from the same articles in order to answers questions about these articles. The representativeness of the articles within all articles depends on the selection criteria of the NewsReader project.

Is the effort required to produce it low?
This depends mostly on how difficult it will be to filter for RelationMentions and to extract triples from these RelationMentions. As with all autogenerated data sets, once the script is written, the level of human effort will be quite low.

Is its quality high?
Because we autogenerate, there are not going to be any accidental human errors, but the overall quality is once again dependent on the quality of the mentions provided by the NewsReader project

Is it well documented?
This is the documentation so far - judge for yourself.

### Further steps required
We stated that our goal was to answer natural language QA pairs. So far we only talked about the generation of incomplete and complete triplets. However, as we limited the style of our questions greatly, the translation to natural language can be done by simply "filling in the blanks". "Who is the mother fo Eddadottir" --> (?, is mother of, Eddadottir) --> (Edda,is mother of, Eddadottir) --> "Edda is the mother of Eddadottir."

## Week 12.11 -18.11

### Data Acquisition
We now have the ability to filter for RelationMentions, which will enable us to explore the data from which we aim to generate the QA pairs for the training of our classifier. However, so far there seems to be no obvious way to translate the data contained in the RelationMentions into QA pairs. An alternative candidate for the generation of QA pairs are the Events associated with each articles, since they seem to contain more semantic information like which actors participated in an action. Our next task will be to generate triples of the form (Subject, Relation, Object) for each article either from the RelationMentions, from the Events or from both.
For the purposes of exploration code to access all RelationMentions and all Events associated with an article has been made available in the file explorer.py.

## Week 19.11 -25.11

### Terminology
There are currently multiple terms for each entity. Some of them are quite confusing, since they are either used by others or by the KnowledgeStore already. Here a few clarifications:
Agent is the active part in a triplet, i.e. "Mr Bigshot" in our example  "Who owns the Company BigCompany?" We previously called this "Subject", but we now use "Agent" to correspond to the use of the terminology in the seminar.
Similarily, Patient is the passive part in a triplet, i.e. "BigCompany" in our example "Who owns the Company BigCompany?" We previously called this Object.
An event that has both an agent and a patient is considered a "Relation". We will continue to use this term, but be aware that there may be events that are not relations. However,we most likely won't be talking about these in our part.

### Current State of the Data
We now have a proper data set consiting of (Agent, ?, Patient) and the raw text of the article as input, with (Agent, Relation, Patient) as ground truth. For now, we generate this once before the actual training/testing from the KnowledgeStore, and save it in a csv file. This process is currently quite time consuming, and requires some parallelization in order to generate the complete data set. We disgregarded the real world knowledge contained about the entities contained within the database, simply because this is easier for our first try, but kept the code modular enough that adding real world relation should be possible without breaking prior models.

### The Code
Via several SPARQL queries we aquire all events that have an agent and a patient (called propbank A0 and A1, respectively) for a given article. We then transform this information to question triplets (Agent, ?, Patient), where a literal "?" represents the value questioned and answer triplets (Agent, Relation, Patient). We then enrich this QA Pair with additional information. Right now, this is only the plain text of the article that was used to generate the pairs, so that the model has any information at all. However, since we used pandas to construct this data set, we were able to construct the data in such a manner that additional information can be added without being available to models that do not request it. This means that we can add additional information at a later date without breaking or changing our prior models.

## Week 26.11 - 02.12

### Data Set Goal Statement
After discussing the minimal requirements for our data set, we adjusted our requirements accordingly. We will now not only generate answers in the form of complete triples and questions in the form of incomplete triples from the articles, but also wrong anwers in order to be able to train our classifier better.

The updated format for our data set is as follows:
We would like to generate a pandas dataframe with the columns "agent", "patient", "word_start_char", "word_end_char", "classification" and "article_uri".
"agent" will contain natural language string values generated from the "propbank:A0" attributes in events.
"patient" will contain natural language string values generated from the "propbank:A1" attributes in events.
"word_start_char" and "word_end_char" will contain int values denoting the position of words within an article.
"classification" will contain boolean values of True or False depending on whether the word denoted by "word_start_char" and "word_end_char" is one of the mentions associated with the corresponding or not.
"article_uri" will contain string values denoting the uri at which the raw text of the article in question can be obtained.

The column "classification" will be the target for our binary classificator.

### Data Set Generation Process
We will use the scripts data_generator.py and explorer.py to generate rows for the pandas dataframe discussed from each article in the table all_article_uris.csv.
For each article, we will generate a triple from each event in the article that has values for the attributes "propbank:A0", "propbank:A1", "rdfs:label" and "gaf:denotedBy".
For each of these triples we will generate a row for each of the words in the article, with the classification being True if and only if the word is one of the mentions denoted by the "gaf:denotedBy" attribute of the corresponding event.
Assuming roughly 20 000 articles with on average 10 useful triples generated from each article and articles containing on average 500 words, we are looking at a data set with about 100 000 000 entries.

Even if we end up having to filter most of them out due to nonsensical "propbank:A0" and "propbank:A1" attributes, this should still be more than enough to train a decent binary classifier.
Of course most of these entries are going to be wrong answers since most words in an article are not the answer to a given question, but wrong answers still provide a good training for classifiers.

### Current State of the Generation Process
For the sake of stability, continuity and error avoidance we are processing only chunks of 50 articles at once and then save the resulting dataframe to a csv file in generated_data/classification_data_chunks.
At a rate of 3 articles per minute the generation of each chunk will take about 16 minutes. The generation of all 400 chunks would consequently take about four to five days of straight computing time. If we could manage to parallelize the process, this would accelerate the process greatly.
Therefore it is an unrealistic expectation that we will have the whole data set up and running by the start of the next seminar session. However, we so far managed to produce 15 data chunks containing the data from the first 750 articles. This should be a good starting point for further exploration and the selection of a suitable classifier. 

## Week 03.12. - 09.12
### Current State of the Data Set
We currently have the classification data from 5000 articles at our disposal. SInce this is about 1/4 of our total data set, we will for the purposes of exploration just assume that the data we have is representative of the whole data set. We will continue working on the generation in the days to come and hope that we will soon have the whole data set from all 19 000 articles. 
Access to the IKW Grid Computing facility could greatly enhance our generation speed, we are working on that. 

For now, the existing data is stored in single csv files with each 1000 articles in zip containers. This is done mostly due to the file size upload limit of GitHub. Fortunately, pandas can read csv files directly from zip containers without extracting them first, so this is not much of an issue.

### Data Set Exploration
For the purposes of exploration, we work within the script explorer.py mostly.
A first examination of the data set has concluded that we have a lot of False classifications, which was to be expected since most words in an article are not the answer to a specific question. 
In the preliminary data set, we have a total of 10 491 601 False classifications and 37 479 True classifcations, giving us only 0.597 % True classifications.

#### Evaluation Metric
Accuracy seems to be the intuitive answer. However, since we are not really interested in whether the classifier misses other correct answers but only want it to find at least one correct answer, another metric that is interesting for us is precision. Precision will tell us how often the answers our question answering system gives us are actually correct, which is the quality we want to optimize.

#### Baseline
Because of the inbalance in classifications, the only suitable baseline for now is "always false". Once we have properly started with the feature engineering, there may be a most predictive feature which will outscore "always false", but for now it is the champion among the baselines. Precision is not really defined for the baseline "always false", since there are no False Positives or True Positives when your classifier is "always false". Accuracy however is 99.64 % on our preliminary data set. This implies that we will have to involve some kind of weighting if we want to get significant differences in evaluations of our classifiers.

#### Split
We have enough data to do 10-Fold Crossvalidation, which we hold to be the most reliable system. We would also prefer it if the triples generated from one article would not all be in one split, since we want the evaluation to be independent of the articles in question. Another point to consider is that since we only have very few True classifications, they should be evenly distributed among the splits so that there is no split with no True classifications to learn from.

Fortunately, all these requirements are fulfilled in skickit-learn's StratifiedShuffleSplit. Accordingly we will use this out-of-the-box solution for now. An application can be seen in the script explorer.py. For the baseline evaluation we do not yet use crossvalidation but only a single split, since the equal share of True/False classifcations is guaranteed by the stratification.


