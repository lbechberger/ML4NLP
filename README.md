# ML4NLP - Epsilon
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Epsilon.

## 1.Introduction


We aim to create a system that is capable of answering questions concerning a single document, e.g. when asked "Who owns the Company BigCompany?" after reading a text about this company, it should be able to answer "Mrs Bigshot". For this we make use of the  KnowledgeStore Database (https://knowledgestore.fbk.eu/), in particular the NewsReader project (http://knowledgestore2.fbk.eu/nwr/wikinews/ui) and the articles contained within.
We quickly realised that a complete QA (Question Answering) system capable of parsing the whole range of complex questions ("Whose fault is the decline of the Euro, according to Mrs Moneybags?") would be beyond our scope. Inspired by the triplets that make up entries in RDF ((x,y,z) in general, but we will use (A R P)), we decided to limit the questions to the style of "What is the relation between Agent A and Patient P?". (A ? P)
Other question that ought to be answerable with this would be "To whom does Agent A have the Relation R?" (A R ?) or "Who has the relation R to Patient P?" (? R A), but we choose to focus on the first kind for the sake of simplicity. If a system would be capable of answering all of these three questions, it could find application in many fields, from an improvement of classical search machines (providing answers directly instead of providing pages that provide answers) or tutoring systems (autogenerating question for a exam on a certain historic text, and correcting the answers automatically), to name a few. 

### Further steps required
We stated that our goal was to answer natural language QA pairs. So far we only talked about the generation of incomplete and complete triplets. However, as we limited the style of our questions greatly, the translation to natural language could be done by simply "filling in the blanks". "Who is the mother of Eddadottir" --> (?, is mother of, Eddadottir) --> (Edda,is mother of, Eddadottir) --> "Edda is the mother of Eddadottir." This however, was beyond the task we set ourselves in this project, so it will not be discussed further.

### On Terminology
There are currently multiple terms for each entity. Some of them are quite confusing, since they are either used by others or by the KnowledgeStore already. Here a few clarifications:
Agent is the active part in a triplet, i.e. "Mr Bigshot" in our example  "Who owns the Company BigCompany?" It also was previously called "Subject", but we  use "Agent" to correspond to the use of the terminology in the seminar.
Similarily, Patient is the passive part in a triplet, i.e. "BigCompany" in our example "Who owns the Company BigCompany?" It  previously was called  Object.
An event that has both an agent and a patient is considered a "Relation". We will use this term, but be aware that there may be events that are not relations. However,we most likely won't be talking about these.


## 2.Data Properties

### Choice of Input and Target
We decided that the target of the predictions ought to be the completed triple of the (Agent Relation Patient) form.
Concerning the input, we decided to consider each word of a text to be a potential answer, (except for stopwords such as "the", "a" etc.), and provide a sliding window of words surrounding the word, together with a number of engineered features that will be explained in their own chapter

### Data Collection
As discussed before, we want to make use of the data contained within the KnowledgeStore database. We desire a large amount of data to enable more complicated machine learning approaches, and we want data clean of formating/labeling issues that could break our pipeline. As such, we decided to autogenerate the data from the DB where possible. 
Researching the database and the entries in the database revealed that there is one kind of entry in the KnowledgeStore database, which is called "Event". Via several SPARQL queries we aquire all events that have an agent and a patient (called propbank A0 and A1, respectively) for a given article. We then transform this information to question triplets (Agent, ?, Patient), where a literal "?" represents the value questioned and answer triplets (Agent, Relation, Patient). (For the purposes of exploration code to access all Events associated with an article has been made available in the file explorer.py.)

###  Pros and Cons
As mentioned, we wanted a large and syntactically clean data set, which is possible with autogeneration. However, we are aware of the limitations. We cannot surpass the semantic quality of the KnowledgeStore, nor can we generate information not yet contained within it, possibly an issue if we want to apply the algorythm to articles outside of the DB. For the articles contained within, however, we are quite sure to be capable of generating representative QA pairs, as these articles are related in their content and style.

### Properties of the Data Set
The size is determined by the size of the NewsReader data set, which currently contains 19 751 articles. Representativeness for our task is given, since we select information from the same articles in order to answers questions about these articles. The representativeness of the articles within all articles depends on the selection criteria of the NewsReader project.
Because we autogenerate, there are not going to be any accidental human errors, but the overall quality is once again dependent on the quality of the mentions provided by the NewsReader project, which we consider to be quite high

### Data Set Generation Process
We will use the scripts data_generator.py and explorer.py to generate rows data from each article in the table all_article_uris.csv.
For each article, we will generate a triple from each event in the article that has values for the attributes "propbank:A0", "propbank:A1", "rdfs:label" and "gaf:denotedBy".
For each of these triples we will generate a row for each of the words in the article, with the classification being True if and only if the word is one of the mentions denoted by the "gaf:denotedBy" attribute of the corresponding event.
Assuming roughly 20 000 articles with on average 10 useful triples generated from each article and articles containing on average 500 words, we are looking at a data set with about 100 000 000 entries.
Of course most of these entries are going to be wrong answers since most words in an article are not the answer to a given question, but wrong answers still provide a good training for classifiers.

## 3.Feature Engineering

### Feature Extraction

It is clear that we have to look at the word under consideration within the context of the words surrounding it - the only question is how big of a neighborhood we want to take into account. A preliminary compromise seems to be looking at the word under consideration only within the context of the words surrounding it - in most of the cases, this is where the critical information is going to be stored. 
Our idea is therefore to extract new features mainly from all the other words , filtered for stopwords using TF-IDF. Since we will further extract features from those surrounding words, a limitation to a small number seems necessary to limit computation time. 

One possible problem with this approach, however, is the loss of information from personal pronouns. For example if we have in an article the sentences "Trump stares angrily at Mueller. He is investigating the president." and the word under consideration is "investigating", agent "Mueller" and patient "Trump", then "investigating" is obviously the correct answer and the completed triple would be (Mueller, investigating, Trump). However, by looking only at the sentence that "investigating" is in, not even an human could infer what was meant without outside knowledge - for a classifier it would be virtually impossible. Our  approach to solving this problem would be the enhancement of the surrounding words using coreference resolution (to get the true meaning of personal pronouns like "him").

### Syntactic Features

One of the more predictive features is be the position of the word in consideration in relation the nearest occurence of the agent and the patient in the text. This alone will be enough to predict a lot of answers, since in the English language the most common syntax is "Subject Predicate Object", where Subject corresponds to agent, Predicate to relation and Object to patient
Other syntactic features that we seek to extract from  all the words in the sentence are POS and NER as well as simple grammatical features (tense, plural/singular, gender).

### Semantic Features
We are very fascinated by the word2vec model, however the computation power required to make use of the semantic information proved prohibitive. We used the  pretrained model provided by Google (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), and while we were able to generate a semantic vector for a part of the data, we were not able to merge it properly with the other features

### Final State
We extracted a 10 word vector around each candidate word, together with the agent and the patient word as strings. For each of these words, we generated a position within the text, as int 16, as well as Part of Speech, Named Entity Recognition and Depedency Category tokens, both saved as int16 values as well. Of these, random forests and wrapper methods showed 'relation_DEP', 'relation_NER', 'relation_POS' to be most promising, which makes sense as these are most closely related to the candidate word, as well as the 'agent_POS' and 'patient_POS' if encoded as distance to the candidate word. 
We also planned to make use of cosinus distances between the agent/patient/relation words' semantic vectors, but merging these results with the rest of the data proves to be an issue due to the size of the generated data (and some data corruption issues).
Currently everything but the binary classification value is kept in int16, as int32, or even worse Strings or mixed Object types cause the size requirements of the dataframes to explode.

## 4. Model Selection

So far, possible models have only briefly been discussed. After some discussion we came to the conlcusion that it would be best to use a classification algorithm to predict for each non-stop word in an article whether or not it completes the triple in question. This approach solves the overall problem of answering questions on a given article not yet completely, but provides a semantic basis for the answering of the question, which would in an application scenario then be further translated into natural language.  

Due to the large size of the data set, we considered only classifiers that fulfill the following requirements:
* Low memory usage
* High speed

In the end, only tree-based ensemble models were able to fulfill these requirements sufficiently. Especially Gradient Boosting and Random Forest classification methods proved to be most effective in predicting the missing word from the triple.
Consequently, in the final evaluation we focused mainly on the following implementations:  

* eXtreme Gradient Boosting (XGB)
* MicrosoftLight Gradient Boosting Machine (LGBM)
* scikit-learn Random Forest Classifier (RF)

In addition to those three, two attempts to further improve performance by using ensembling methods on these three implementations were made.
For this purpose, the following implementations of ensembling methods were used:  

* mlens SuperLearner
* scikit-learn Voter

In the following we provide a short explanation for some of the lesser known methods,


### Gradient Boosting

Gradient boosting is a common machine learning method for classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model stage-wise and generalizes them by allowing optimization of an arbitrary differentiable loss function.  

The main difference between the two used implementations (XGB and LGBM) is that while XGB uses an pre-sort-based algorithm for decision tree learning, LGBM uses a histogram-based algorithm that buckets continuous feature values into discrete bins. This speeds up training and reduces memory usage.

### Stacking

Stacking is a common ensembling method that uses a number of existing models as base learners, whose predictions are then used by a so-called super learner to make its own predictions. This can sometimes super learner can sometimes achieve higher scores than the best base learner by learning to decide in which cases which base learner scores the best results. The implementation that we used was provided by the mlens package.

### Voting

Voting is a common ensembling method that like Stacking uses a number of existing models as base learners and makes its own predictions based on their predictions. A voting classifier is however much less sophisticated and makes decisions simply by majority vote. There are two types of voting classifiers provided by scikit-learn, hard and soft voting. Hard voting means using the binary predictions of the base learns while soft voting means using the predicted probabilities to make decisions. In the following, when we speak of model V, we mean a soft voting classifier, since hard voting turned out to produce much worse results on all metrics.


## 5. Evaluation Procedure

Three versions of our data set were compiled for the purposes of model evaluation:

* Heavy:  
The full data set that we assembled during the data generation phase - position, POS, DEP and NEC for agent, patient, relation words and all words in a sliding window from -10 to +10 non-stop words around the relation word for up to 2 sentences
* Light:  
A reduced version of the data set - position, POS, DEP and NEC only for agent, patient and relation words, DEP only for a sliding window from -7 to +7 stop words
* Hot:  
A minimal version of the data set - distance of patient and agent position to relation position and only features with a feature importance of more than 0.01 in all three implementations from a hot-encoded version of POS, DEP and NEC of the relation word

The initial exploration of the different models and parameters was done almost exclusively on the Hot data set, since it provided the fastest results due to its low file size and low number of features. For the fine tuning and final evaluation, however, we moved on to the Light data set, since it is more complete and easier to replicate in an application scenario. The Heavy data set was sadly never really put to use due to its large file size and high number of features which caused the runtime to exceed acceptable dimensions on the available hardware.  

For all mentioned transformations and other utilites we used, unless specified otherwise, the implementations provided by the scikit-learn.  

In order to split the data set into training and test data for the purposes of cross-valdiation, a instance of StratifiedShuffleSplit was used. Its properties are prefectly suited to our data set. It shuffles the data to ensure that a wide range of articles are being used while at the same time preserving the overall ratio of positive and negative samples.  

The models were evaluated mainly by two metrics: Precision and Recall. Accuracy - perhaps the more conventional choice - was out of the question, since our data set contains far more negative than positive samples and thus a naive "always false" classifier would already have scored an accuracy of > 0.95, which is not representative of the success.  

The tuning of the hyperparameters for the selected classifiers was done via GridSearchCV. At first the parameter grid was distributed widely to explore the full range of possible combinations. Then the grid was tightened around parameters that returned the highest scores so far.  

## 6. Results

The following results were obtained by using the best parameters found using grid search for each model and scoring over 10 splits with 90% training data and 10% test data each.  

### Evaluation results on the Hot data set

<table class="js-csv-data csv-data js-file-line-container">
      <thead>
        <tr id="LC1" class="js-file-line">
          <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
            <th></th>
            <th>model</th>
            <th>average precision</th>
            <th>average recall</th>
        </tr>
      </thead>
      <tbody>
          <tr id="LC2" class="js-file-line">
            <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
              <td>0</td>
              <td>XGB</td>
              <td>0.71985</td>
              <td>0.15620</td>
          </tr>
          <tr id="LC3" class="js-file-line">
            <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
              <td>1</td>
              <td>LGBM</td>
              <td>0.74871</td>
              <td>0.12602</td>
          </tr>
          <tr id="LC4" class="js-file-line">
            <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
              <td>2</td>
              <td>RF</td>
              <td>0.68972</td>
              <td>0.18481</td>
          </tr>
          <tr id="LC5" class="js-file-line">
            <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
              <td>3</td>
              <td>SL</td>
              <td>0.69362</td>
              <td>0.12403</td>
          </tr>
	  <tr id="LC6" class="js-file-line">
            <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
              <td>4</td>
              <td>V</td>
              <td>0.72419</td>
              <td>0.15972</td>
          </tr>
    </tbody>
</table>

### Evaluation results on the Light data set

<table>
      <thead>
        <tr id="LC1" class="js-file-line">
          <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
            <th></th>
            <th>model</th>
            <th>average precision</th>
            <th>average recall</th>
        </tr>
      </thead>
      <tbody>
          <tr id="LC2" class="js-file-line">
            <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
              <td>0</td>
              <td>XGB</td>
              <td>0.73712</td>
              <td>0.13123</td>
          </tr>
          <tr id="LC3" class="js-file-line">
            <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
              <td>1</td>
              <td>LGBM</td>
              <td>0.79055</td>
              <td>0.10212</td>
          </tr>
          <tr id="LC4" class="js-file-line">
            <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
              <td>2</td>
              <td>RF</td>
              <td>0.58057</td>
              <td>0.13989</td>
          </tr>
          <tr id="LC5" class="js-file-line">
            <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
              <td>3</td>
              <td>SL</td>
              <td>0.64781</td>
              <td>0.12948</td>
          </tr>
	  <tr id="LC6" class="js-file-line">
            <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
              <td>4</td>
              <td>V</td>
              <td>0.63876</td>
              <td>0.14764</td>
          </tr>
    </tbody>
</table>

### Best Parameters

XGBoostClassifier:
* "n_estimators": 64,
* "objective": "binary:logistic",
* "max_depth": 3,
* "min_child_weight": 1,
* "subsample": 0.5,
* "colsample_bytree": 0.5,
* "learning_rate": 0.12,
* "tree_method": "exact",
* "reg_alpha": 0.6,
* "reg_lambda": 0.6

LGBMClassifier:
* "n_estimators": 256,
* "max_depth": 7,
* "num_leaves": 512,
* "objective": "binary",
* "min_child_samples": 20,
* "reg_alpha": 0.4,
* "reg_lambda": 0.4,
* "learning_rate": 0.005

RandomForestClassifier:
* "n_estimators": 64,
* "criterion": "gini",
* "max_depth": 12,
* "max_features": "log2",
* "min_samples_split": 4

### Prediction Samples

<table>
  <tr>
    <th>agent</th>
    <th>patient</th>
    <th>relation</th>
    <th>truth</th>
    <th>predicted</th>
  </tr>
  <tr>
    <td>the group</td>
    <td>a press release</td>
    <td>sent</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>bomb disposal</td>
    <td>two and a half hours</td>
    <td>took</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>George W. Bush</td>
    <td>a speech</td>
    <td>started</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Mustafa Kemal Atatürk</td>
    <td>the country</td>
    <td>founded</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Mitt Romney</td>
    <td>the required 1144 delegates</td>
    <td>surpassed</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Boston Red Sox</td>
    <td>the world series championship</td>
    <td>dominated</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Thomas Jefferson</td>
    <td>a maximum sentence of 235 years</td>
    <td>faces</td>
    <td>1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>islamic insurgents</td>
    <td>three people</td>
    <td>shot</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>their bombardment</td>
    <td>the lives of UN staff</td>
    <td>endangered</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Zinedine Zindane</td>
    <td>Marco Materazzi</td>
    <td>headbutted</td>
    <td>1</td>
    <td>1</td>
  </tr>
</table>

### Summary

As can be seen in the above tables, the highest average precision was achieved by an LGBM model on the Light data set with a score of 0.79055.
The highest average recall, however, was achieved by a Random Forest Classifier on the Hot data set with a score of 0.18481.  

In general, we observed that there seemed to be a trade-off between recall and precision. To increase one always meant to decrease the other. The Random Forest implementation seemed in general to be better able to maximize recall, while the gradient boosting methods scored better in precision. In terms of speed, LGBM is the unquestionable winner - it always took no more than half the time that the other implementations needed while often at the same time scoring much higher in precision.  

Some parameters such as reg_alpha and reg_lambda in the gradient boosting methods simply improved performance in both metrics. Other parameters, however, influenced the balance between recall and precision heavily. A high learning rate for example increased recall in all cases while a high number of estimators usually decreased it.  

The ensembling methods Stacking and Voting did not prove to be effective. In fact, they seemed to combine the worst from all models, decreasing both recall and precision while also taking a lot of time.  

## 7. Discussion

One possible reason for the ineffectiveness of the ensemble methods in this case could be that the base models were already ensemble methods of their own. One could argue that the models themselves were already optimal combinations of weaker models, and further combing these models with other models destroyed the delicate balance that was ingrained in them.  

 
