# ML4NLP - Epsilon
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Epsilon.

## Introduction


We aim to create a system that is capable of answering questions concerning a single document, e.g. when asked "Who owns the Company BigCompany?" after reading a text about this company, it should be able to answer "Mrs Bigshot". For this we make use of the  KnowledgeStore Database (https://knowledgestore.fbk.eu/), in particular the NewsReader project (http://knowledgestore2.fbk.eu/nwr/wikinews/ui) and the articles contained within.
We quickly realised that a complete QA (Question Answering) system capable of parsing the whole range of complex questions ("Whose fault is the decline of the Euro, according to Mrs Moneybags?") would be beyond our scope. Inspired by the triplets that make up entries in RDF ((x,y,z) in general, but we will use (A R P)), we decided to limit the questions to the style of "What is the relation between Agent A and Patient P?". (A ? P)
Other question that ought to be answerable with this would be "To whom does Agent A have the Relation R?" (A R ?) or "Who has the relation R to Patient P?" (? R A), but we choose to focus on the first kind for the sake of simplicity. If a system would be capable of answering all of these three questions, it could find application in many fields, from an improvement of classical search machines (providing answers directly instead of providing pages that provide answers) or tutoring systems (autogenerating question for a exam on a certain historic text, and correcting the answers automatically), to name a few. 

### Further steps required
We stated that our goal was to answer natural language QA pairs. So far we only talked about the generation of incomplete and complete triplets. However, as we limited the style of our questions greatly, the translation to natural language could be done by simply "filling in the blanks". "Who is the mother of Eddadottir" --> (?, is mother of, Eddadottir) --> (Edda,is mother of, Eddadottir) --> "Edda is the mother of Eddadottir." This however, was beyond the task we set ourselves in this project, so it will not be discussed further.

### On Terminology
There are currently multiple terms for each entity. Some of them are quite confusing, since they are either used by others or by the KnowledgeStore already. Here a few clarifications:
Agent is the active part in a triplet, i.e. "Mr Bigshot" in our example  "Who owns the Company BigCompany?" It also was previously called this "Subject", but we  use "Agent" to correspond to the use of the terminology in the seminar.
Similarily, Patient is the passive part in a triplet, i.e. "BigCompany" in our example "Who owns the Company BigCompany?" It  previously was called  Object.
An event that has both an agent and a patient is considered a "Relation". We will use this term, but be aware that there may be events that are not relations. However,we most likely won't be talking about these.


## Data Properties

### Choice of Input and Target
We decided that the target of the predictions ought to be the completed triple of the (Agent Relation Patient) form.
Concerning the input, we decided to consider each word of a text to be a potential answer, (except for stopwords such as "the", "a" etc.), and provide a sliding window of words surrounding the word, together with a number of engineered features that will be explained in their own chapter

### Data Collection
As discussed before, we want to make use of the data contained within the KnowledgeStore database. We desire a large amount of data to enable more complicated machine learning approaches, and we want data clean of formating/labeling issues that could break our pipeline. As such, we decided to autogenerate the data from the DB where possible. 
Researching the database and the entries in the database revealed that there is one kind of entry in the KnowledgeStore database, which is called "Event". Via several SPARQL queries we aquire all events that have an agent and a patient (called propbank A0 and A1, respectively) for a given article. We then transform this information to question triplets (Agent, ?, Patient), where a literal "?" represents the value questioned and answer triplets (Agent, Relation, Patient). (For the purposes of exploration code to access all Events associated with an article has been made available in the file explorer.py.)

###  Pros and Cons
As mentioned, we wanted a large and syntactically clean data set, which is possible with autogeneration. However, we are aware of the limitations. We cannot surpass the semantic quality of the KnowledgeStore, nor can we generate information not yet contained within it, possibly an issue if we want to apply the algorythm to articles outside of the DB. For the articles contained within, however, we are quite sure to be capable of generating representative QA pairs, as these articles are related in their content and style.

### Properties of the Data Set
The size is determined by the size of the NewsReader data set, which currently contains 19751 articles. Representativeness for our task is given, since we select information from the same articles in order to answers questions about these articles. The representativeness of the articles within all articles depends on the selection criteria of the NewsReader project.
Because we autogenerate, there are not going to be any accidental human errors, but the overall quality is once again dependent on the quality of the mentions provided by the NewsReader project, which we consider to be quite high

### Data Set Generation Process
We will use the scripts data_generator.py and explorer.py to generate rows data from each article in the table all_article_uris.csv.
For each article, we will generate a triple from each event in the article that has values for the attributes "propbank:A0", "propbank:A1", "rdfs:label" and "gaf:denotedBy".
For each of these triples we will generate a row for each of the words in the article, with the classification being True if and only if the word is one of the mentions denoted by the "gaf:denotedBy" attribute of the corresponding event.
Assuming roughly 20 000 articles with on average 10 useful triples generated from each article and articles containing on average 500 words, we are looking at a data set with about 100 000 000 entries.
Of course most of these entries are going to be wrong answers since most words in an article are not the answer to a given question, but wrong answers still provide a good training for classifiers.

## Feature Engineering

### Feature Extraction

It is clear that we have to look at the word under consideration within the context of the words surrounding it - the only question is how big of a neighborhood we want to take into account. A preliminary compromise seems to be looking at the word under consideration only within the context of the words surrounding it - in most of the cases, this is where the critical information is going to be stored. 
Our idea is therefore to extract new features mainly from all the other words , filtered for stopwords using TF-IDF. Since we will further extract features from those surrounding words, a limitation to a small number seems necessary to limit computation time. 

One possible problem with this approach, however, is the loss of information from personal pronouns. For example if we have in an article the sentences "Trump stares angrily at Mueller. He is investigating the president." and the word under consideration is "investigating", agent "Mueller" and patient "Trump", then "investigating" is obviously the correct answer and the completed triple would be (Mueller, investigating, Trump). However, by looking only at the sentence that "investigating" is in, not even an human could infer what was meant without outside knowledge - for a classifier it would be virtually impossible. Our  approach to solving this problem would be the enhancement of the surrounding words using coreference resolution (to get the true meaning of personal pronouns like "him").

### Syntactic Features

One of the more predictive features is be the position of the word in consideration in relation the nearest occurence of the agent and the patient in the text. This alone will be enough to predict a lot of answers, since in the English language the most common syntax is "Subject Predicate Object", where Subject corresponds to agent, Predicate to relation and Object to patient
Other syntactic features that we seek to extract from  all the words in the sentence are POS and NER as well as simple grammatical features (tense, plural/singular, gender).

### Semantic Features
We are very fascinated by the word2vec model, however the computation power required to make use of the semantic information proved prohibitive. We used the  pretrained model provided by Google (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), and while we were able to generate a semantic vector for a part of the data, we were not able to merge it properly with the other features

## Final State
We extracted a 10 word vector around each candidate word, together with the agent and the patient word as strings. For each of these words, we generated a position within the text, as int 16, as well as Part of Speech, Named Entity Recognition and Depedency Category tokens, both saved as int16 values as well. Of these, random forests and wrapper methods showed 'relation_DEP', 'relation_NER', 'relation_POS' to be most promising, which makes sense as these are most closely related to the candidate word, as well as the 'agent_POS' and 'patient_POS' if encoded as distance to the candidate word. 
We also planned to make use of cosinus distances between the agent/patient/relation words' semantic vectors, but merging these results with the rest of the data proves to be an issue due to the size of the generated data (and some data corruption issues).
Currently everything but the binary classification value is kept in int16, as int32, or even worse Strings or mixed Object types cause the size requirements of the dataframes to explode.

## Choice of Model

### Choice of Model
So far, possible models have only briefly been discussed. One possible choice would be the use of a classification algorithm which predicts for each word or semantic unit in the text whether or not it matches the missing part of the triple. This solves the overall problem of answering questions on a given article not yet completely, but provides a semantic basis for the answering of the question. When training with the QAP database, the output of the classification algorithm will have to be translated into the denominator of the corresponding node in the database. When working with a natural language question as input, the completed triple should ideally be further translated into a natural language answer by means of natural language generation.

## Evaluation
### Evaluation Metric
Accuracy seems to be the intuitive answer. However, since we are not really interested in whether the classifier misses other correct answers but only want it to find at least one correct answer, another metric that is interesting for us is precision. Precision will tell us how often the answers our question answering system gives us are actually correct, which is the quality we want to optimize.

### Baseline
Because of the inbalance in classifications, the only suitable baseline for now is "always false". Once we have properly started with the feature engineering, there may be a most predictive feature which will outscore "always false", but for now it is the champion among the baselines. Precision is not really defined for the baseline "always false", since there are no False Positives or True Positives when your classifier is "always false". Accuracy however is 99.64 % on our preliminary data set. This implies that we will have to involve some kind of weighting if we want to get significant differences in evaluations of our classifiers.

### Split
We have enough data to do 10-Fold Crossvalidation, which we hold to be the most reliable system. We would also prefer it if the triples generated from one article would not all be in one split, since we want the evaluation to be independent of the articles in question. Another point to consider is that since we only have very few True classifications, they should be evenly distributed among the splits so that there is no split with no True classifications to learn from.

Fortunately, all these requirements are fulfilled in scikit-learn's StratifiedShuffleSplit. Accordingly we will use this out-of-the-box solution for now. An application can be seen in the script explorer.py. For the baseline evaluation we do not yet use crossvalidation but only a single split, since the equal share of True/False classifcations is guaranteed by the stratification.

## Discussion


## New Repository
Due to problems with this repository, we created a new repository, where we will do most of the classificator configuration. For now we have a "light" dataset, which amounts to about 1.1 GB unzipped and includes only the following features:
relation_NER, relation_DEP, relation_POS, agent_position, patient_position (both relative to relation_positon) and DEP for words in sliding window of -7 to +7.
The "heavy" dataset amounts to 2.6 GB unzipped and includes additionally the positions of all words in the sliding window and extends it to -10/+10.
We have not been able to implement semantic vectors and synonyms yet and are for now working with the light version to configure classificators.
You can find our new repository here: https://github.com/Nathanaelion/NRRC
 
