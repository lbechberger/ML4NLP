# ML4NLP - Delta
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Delta.

# 4.11.2018

### Overall task

The overall task of this work is to create a Question-Answering-System, that is able to answer simple user-questions about documents as for example news pages. For this, we make use of the KnowledgeStore Database (https://knowledgestore.fbk.eu/). As part of this database, the NewsReader project (http://www.newsreader-project.eu/) built up a database of knowledge-triples in the RDF-format from news articles, which contain information triples extracted from these articles. These information-triples in combination with the articles are supposed to be our training data. Using this data has implications on the scope of our task - while a general QA-system is in itself a fairly hard task to solve, breaking our problem down into questions that can be answered using information-triples, only information that is part of such a triple can be queried.

#### Answerable Questions

All triples provided as part of the KnowledgeStore-database (http://knowledgestore2.fbk.eu/nwr/wikinews/ui) are of the form *subject-predicate-object*. This limits the scope of questions we can answer to the following three general types, asking respectively for the subject, the predicate/relation, and the object:

- **Who**  \<did something\> to \<some entity\>? (For example "Who shot Abraham Lincoln?")
- **What** did \<some entity\> do to \<some entity\>? (For example "What did John Wilkes Booth to Abraham Lincoln?")
- **To whom** did \<some entity\> do \<something\>? (For example "Whom did John Wilkes Booth shoot?")

Using this data has the huge advantage that we don't have to manually create a dataset by extracting Question-Answer-Pairs from news articles ourselves. The only task left to create the dataset is thus to extract all triples in a way that's useful as dataset for our task, one idea being to create a dataset of pairs (one component missing) together with the underlying article as input, and the missing component as target.

The overall aim of this project is that it can, after successful training, be used to ask the specified types of questions on any kind of news article, even ones where no information was crawled and made available in a semantic database like the KnowledgeStoreDB.

### How it's broken down into a classification problem

Breaking this task down into a classification problem will most likely be the biggest challenge of this task. As per demand of the lecturer, the entire task must be solved by classifications. While this is not only unintuitive for a quesiton-anwering-system per se, it also vastly limits the scope of our models. While modern sequence-to-sequence models relying on deep neural networks achieve incredible performance on publicly availabe QA-datasets (see eg. https://rajpurkar.github.io/SQuAD-explorer/, where the best model is at 74%, just 12pps short of human performance), these models are obviously not classificators.

When going through this pipeline not in an end-to-end-fashion, but step-by-step, there are several components that must be tackled:

#### Extracting the demanded triple from a question

The first problem arises already when posing the question. As questions can be asked in almost infinitely different ways, this may already be a candidate to use machine learning for. As we however do not have the possibility to easily generate a dataset fot this, we will most likely develop other ways to do so, incorporating solutions as for example Watson's *lexical answer type detection* (see Building Watson: An Overview of the DeepQA Project[1])

#### Selecting the correct article that answers the question

After the respective two-thirds-of-a-triple that constitutes the question is successfully extracted (eg. from "Who shot Abraham Lincoln" to "\<X\> \<shot\> \<Lincoln\>") the next challenge is to select the article that answers this question. As all triples in the KnowledgeStore-database contain the respective source article, it seems possible to generate a dataset for a model that is able to figure out this connection. This model would then have as classes a vector of all eligible articles, selecting the one that is relevant for the demanded tuple. This is, of course, only a rough idea, and many questions (as for example the encoding of the article and the tuple, if they are supposed to be bags-of-words, or already contain meta-information as URIs of the KnowledgeStore-database-entities) are still open. Further, it is an open question of how this step may be performed on an ever-changing database of articles, once the model runs productively.

#### Answering the question given the correct article

After the article that may be able to answer the question is selected, the question must be answered using this article. For this, a seperate model must most likely be developed (While a general rule-of-thumb in machine learning is that end-to-end-architectures work better than when split up into their compoments, it would constitute a far-too-huge problem if using the complete database of articles). This model is trained on a dataset of incomplete knowledge-triples and the corresponding article. A possible idea of what this model may do is that it uses the number of the word that is the missing component of the question-tuple in the article as target, the classes of this model would thus be the range of the number of all words in this article. Using this model, a bag-of-words-approach is however impossible to use, as the information about order in the text must in any case be preserved.

Depending on the performance of such a model, it may even be a good idea to split up this task, depending on which part of the triple (*subject-predicate-object*) is the one which is asked for. If seperate classificators are used for each of these tasks, it would be a perfect oppertunity to split up this task (as well as the corresponding dataset generation) to the three groups which are working on question-answering in this seminar.

The candidate answer (or candidate answers) can then be selected (or ranked by further post-processing, using a similar strategy like [1]), and finally be returned as answer to the question, after the last step:

#### Responding to the question of the user

After the candidate answer is found, it must be put into a complete sentence that constitutes an answer to the question posed by the user. This appears to be a relatively simple task for now however, as simple string-manipulation-techniques can be applied to the questions, especially since the questions are generally posed in the same form (see above).

### Choice of model

This pipeline solves our overall task in four steps, two of which involve using classifiers trained by machine learning. As for the choice of model, no final decision has been made. As the required data is vastly high-dimensional (using bag-of-words- or highdimensional vector-space models), Neural Networks are an obvious choice, in contrast to simple rule-based classifiers. As we however only deal with classification and the size of our dataset may be too small for deep neural networks, other models as support vector machines or random decision forests are also an interesting choice to be looked at. The final decision of the best model will be further discussed after a feasible dataset was created.

With the demanded constraints (namely that it has to be a classificator and that the information-triples of the KnowledgeStoreDB have to be used), we think this pipeline is the best approach to solve the overall task.


## References

[1] A. Ferrucci, David & Brown, Eric & Chu-Carroll, Jennifer & Fan, James & Gondek, David & Kalyanpur, Aditya & Lally, Adam & Murdock, J William & Nyberg, Eric & Prager, John & Schlaefer, Nico & Welty, Christopher. (2010). Building Watson: An Overview of the DeepQA Project. AI Magazine. 31. 59-79.



# 11.11.2018

### Overall task

This week's task is to have the first idea on the labels autogeneration in an elegant manner, which means the autogenerated labels should be representative, low-effort, high-quality, and well documented. As described in last week's documentation, the labels we need are **information-triples** which can be queried in KnowledgeStore Database.
Our **information-triples** include three components, namely two entities and one relation (between two entities).

### General Workflow 1.0

Goal of this subtask (autogeneration of labels) is to generate **information-triples**, which means two entities and one relation (between entities). This can be further broken down into two subtasks.

1. Entities extraction

one possible way to detect entities is to use pre-existing Name Entity Recognition (NER) database (CoNLL 2003), which will be used to retrieve more meta-information of every sentense in our dataset, in order to better autogenerate **information-triples**. Bi-LSTM could be used in this benchmark to achieve relatively good performance (precision/recall/F score at around 90% [1]). Output of this subtask is the following four items (word, POS tag, chunk tag, NER tag) for each word in the article: the first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. For example,

* U.N. NNP I-NP I-ORG
* official NN I-NP O
* Ekeus NNP I-NP I-PER
* heads VBZ I-VP O
* for IN I-PP O
* Baghdad NNP I-NP I-LOC

with this we can extract entities for each sentense.

2. Relation extraction

with the class **nwr:RelationMention** in Knowledgestore API, it is of low cost to extract the relation between two entities. Output of this would be one of the following four different types of relations.

* nwr:TLink - A temporal link, i.e., a mention denoting a temporal relation among two events and/or time expressions.
* nwr:CLink - A causal link, i.e., a mention denoting a causal relation among two events.
* nwr:Participation - A mention denoting the participation of an object (e.g., a person) to a certain event, further characterized by the role played by that object.
* nwr:GLink - A grammatical link among event mentions.
* nwr:SLink - A structural link, i.e., a mention denoting a structural relation among two events.


### About NER database CoNLL 2003

This dataset [1] includes 1,393 English and 909 German news articles, which is compatible with our dataset. We can use the english version for free, fitting our cheapness criterion. Our goal in this task is to extract entities for each sentence. Entities are annotated with LOC (location), ORG (organisation), PER (person) and MISC (miscellaneous). For instance, "**John Wilkes Booth shot Abraham Lincoln**" will be annotated with "**John[PER] Wilkes[PER] Booth[PER] shot[O] Abraham[PER] Lincoln[PER]**". You might notice that **John Wilkes Booth** will be annotated with three entity tags, rather than one single tag. How to overcome this problem needs more investigation.

Some ANN-based models are able to achieve around 90% precision, recall and F score in this benchmark, such as [2]. The choice of architecture still remains to be explored. But Bi-LSTMs could be a good choice, with GloVe or GoogleNews Word2vec word embeddings.

### desiderata

One biggest advantage of our approach is that it can generate entities for each sentense, which can be further combined with Knowledgestore method to extract the relation between entities.

- large: for each **possible** sentence in every article the information-triples will be extracted. To this extent, we are reaching the maximal possibility to get a large data.
- low-effort: absolutely. it's for free and there is an open-source implementation which achieves the current state-of-the-art performance (https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL).
- representative: we limit our scope into tackling answerable questions. Under this constraint, our approach of extracting entities and relation seems to be representative.
- high-quality: Because the NER task will introduce some unavoidable loss, so the quality of labels can not reach to optimal level. However, 90% F score seems to be good enough to keep a trade-off between quality and effort. After extracting information-triples, some manual checks are also expected.
- well-documented: absolutely. With task of NER we will receive a part-of-speech (POS) tag, a syntactic chunk tag and a named entity tag in word-level. Further combined with Knowledgestore API we can retrieve more meta-data.

### Limitation

1. step-by-step fashion of breaking down the problem will accumulate loss.
2. Training for NER requires huge computation power and is time-consuming. (the most power machine we are able to access is a 12G Titan X in IKW-Grid, which is still runnable with reasonable parameters of the ANN model.)


## References

[1] Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition." Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4. Association for Computational Linguistics, 2003.

[2] Chiu, Jason PC, and Eric Nichols. "Named entity recognition with bidirectional LSTM-CNNs." arXiv preprint arXiv:1511.08308 (2015).


# 18.11.2018

### Overall task

This week's task is to have the first code to retrieve data. As discussed in the class, nltk will first be used in retrieving name entities. Our goal remains to be extracting **inforrmation-triples**(two entities and one relation). It's intuitive to use NER function in nltk. Further, nltk also supports detecting pre-defined relation, for instance, *OF* and *IN*, from two pre-defined entitied, for instance, *ORG*(organization), *PER*(person) and *LOC*(location). It is of great help to have a first glimpse on the results from nltk and further decide how to use this information for our task.

### What our code does

We have our first code trying to retrive the **inforrmation-triples** (see *third_week_project.py*). We limit our scope in this first attempt as follows:

1. retrieve two entities

   Four entities are included in our code: *ORG(orrganization), PER(person), LOC(location)* and *GPE(geo-political entities)*.

2. retrieve relation

   Two relations are included: *of* and *in*.

*nltk.sem.relextract.extract_rels* is used in retrieving these two entities and relation. The idea is, manually define the combination of two entities and one relation and scan through a number of articles from *all_article_uris.py* and output a list of entitles-relation. These entitles-relation are auto-generated by nltk. Thus the quality can not be guaranteed. Input of the code is simply number of articles you want to get these entitles-relation from, output is a list of entitles-relations.


### Output

some examples of the output.

1. org_of_gpe
   [ORG: 'Association/NNP'] 'of/IN' [GPE: 'Zoos/NNP']
2. per_of_gpe
   [PER: 'Vladimir/NNP Voronin/NNP'] 'of/IN' [GPE: 'Moldova/NNP']
3. per_in_gpe
   [PER: 'Reverend/NNP Al/NNP Sharpton/NNP'] 'in/IN' [GPE: 'Washington/NNP']
4. per_in_gpe
   [PER: 'Mutambara/NNP'] 'was/VBD arrested/VBN at/IN his/PRP$ house/NN in/IN' [GPE: 'Harare/NNP']

As shown in some of the outputs, the result is not really ideal for the **direct** extraction of the information-triples. However, it could help simplying complex sentences. For example, *Vladimir Voronin in Moldova* is extracted from the original text "President Vladimir Voronin of Moldova today signed a decree nominating ...". Although nltk doesn't do a good job getting the triples, it does a good job getting one element of the triples (the first entity) from a complex sentence (President Vladimir Voronin of Moldova today signed a decree). How to better retrieve all three elements still needed a deeper investigation.

How to store our dataset? Our idea is to directly annotate the output of NER and relation between entities in the original articles, in order to directly feed into our binary classifier in the future. However, current progress doesn't allow us to dream too far away, because a concrete and better way of extracting information-triples needs to be realized and the choice of model for classification needs to be determined before we decide how we want the input to be like.



# 25.11.2018

### State of the code

So far, our code is able to extract certain kinds of relations using Python's NLTK. To do so, we access every article and run some components of the NLTK pipeline over it, in order to extract named entities as well as their relation. As, unfortunately, the necessary components from NLTK in part have a very low accuracy (or recall), only a subset of the possible relations are actually extracted, which is why we are looking into other possibilities to do so. Once we found other ways of doing so, we can simply replace the corresponding NLTK components thanks to our modular codebase.

Another disadvantage about our approach is the fact that we have to define the relations between entities ourselves, using a custom gazetteer. Because of that, by far not all possible relations are actually extracted, and the size of our dataset depends in large parts on manual work of adding more relations by finding the corresponding regular expressions (see above).

These relations are stored using simple CSV-files, where the entire relation is stored, together with the type of named entity and the position where the sentence making up the information appeared ("John Wilkes Booth, PER, shot, Abraham Lincoln, PER, <link>, <number>").

### Relation extraction

After identifying named entities in the articles using NLTK (or, hopefully in the future other techniques), the relations between specified types of named entities are extracted. To do so, we look for all triples of the form (NE, text, NE), and use regular expressions to figure out if the text in between two named entities contains relevant information. As mentioned above, we use regular expressions looking for strings that contain certain kinds of relations (like *in* or *of*). Unfortunaltey, doing this extraction has a relatively low precision, as it also retreives many false positives and inaccuracies: because we simply extract *any* string containing an *in* for example, we cannot tell if a person was born in a place, or if simply an event happened at a place, where that person was not necessarily physically present. Also the named entity recognition is far from perfect. An example where the named entity recognition failed is the following:

`[PER: 'Sea/NNP Launch/NNP'] ',/, a/DT partnership/NN between/IN companies/NNS in/IN' [GPE: 'Norway/NNP']`

Which becomes the relation "Sea Launch, is_in, "Norway", saved as "Sea Launch, PER, is_in, Norway, GPE". As can be seen, the named entity recognition mistakes sea launch for a human.


## 02.12.2018

### State of the code
Our code is able to get triples information *[agent, predicate, patient]* (for convenience, we adapt the words - agent and patient, from the other QA group) for each article with the use of Knowledgestore Databse. Computation is done in IKW grid.

### Workflow
Given an article, all mentions in the article are retrieved. Then we find all mentions with a predicate. Among those mentions with predicates, we further limit our search with a sparql query to retrieve all events with a patient, a predicate and a patient. The motivation of first finding predicates is that it gives more constraints to the SPARQL query of finding events with agents and patients, so that the computation time for the task will be reduced.
![workflow](/figures/week6.png)

### Change in strategy
one change in our workflow is that we now first find the relation(predicate) between entities, rather than retrieving relation after retrrieving entities. Reason is as explained above, to reduce computation time.

### Result
we have totally 19,751 news articles. To speed up, we distribute the job to 20 different machines, with each machine 1000 articles. And every machine will write the result in a separate csv file in folder *data/raw_csv*. This allows us to retrieve information from all the articles in less than 8 hours, ideally. However, due to memory reason, some jobs are killed after a certain time period. With this acknowledgement, we save the triples information to csv after every article. This allow us to 1) check which articles are missing efficiently; 2) to resume the jobs after they are killed easily.

In our first result, information from 15,138 articles are retrieved and average computing time for 1000 articles is 7.5 hours. 4,613 are still missing due to memory problem in the IKW grid. Current average number of triples per article extracted by our strategy is 9. 139,343 triples are found from 15,138 articles. (the missing articles are currently running in the grid. This will be updated later.)

==============================update==============================
=======
# 09.12.2018
### State of the Dataset
Current state of the dataset generate triples solely based on the retrieval algorithm from Sparql. We have total number of 172,758 triples from all news articles in *all_article_uris.csv*. However, this dataset still contains lots of noise. To clean up the dataset and get more meaningful result, several approaches are experimented. We have looked at the *altnames* attribute in the *entity layer* for both agents and patients, which are retrieved with *propbank:A0* and *propbank:A1*. The parsing algorithm in Sparql does not seem to perform well, with many false coreference to simple names (eg., John) and pronouns (eg., He or She or the man). So we gave up this approach in order not to further pollute our dataset. Another approach we proposed with group Zeta is to use their nltk-based code to work on our already existing triples dataset, and to take only the samples that are classified both by zeta's nltk-based extractor and by our extractor. Idea is to filter the samples we have so far, such that we only find realistic ones. We will discuss this approach in more details with group zeta in the next session.

### Explanation on our codes
we have three main files for data autogeneration.

1. generate_data.py

this is to auto-generate triples from news articles stored in *all_article_uris.csv* based on Sparql command.

2. generate_data.sge

this is to run the parallelism on IKW Grid.

3. entities_meta_information.py

this is to retrieve meta-information of the entity layer such as alternative names, birth place/data, gender of a given entity, for explorative use.

### Some more collaboration with group Zeta
First, we collaborate on triples extraction as explained above.

Second, we collaborate on negative samples generation. Current idea is to substitute the true *agents* and *patients* entities with other random ones (different from the real ones) from all the agents and patients we have in our dataset, using group zeta's *negatives_generation.py*. This guarantees we can have as many negative examples as we want. However, the quality of negatives example is difficult to measure. Alternative approach is to generate negative examples with entities only from the same article, which also greatly limits the number of negative examples generated. For this, we will also need to have further discussion with group Zeta in the next session. (examples of negative examples are in *negatives.csv*)

### Split up the data
We decided to split the dataset 50/50 such that we don't have any bias. This can be done after the discussion with group Zeta. For training, 10-fold cross validation would be considered. With the computation power of IKW Grid, this should not be a problem to run.

### Evaluation
For evaluation, we use either cohen's kappa or matthews correlation coefficient (MCC) as evaluation metrics. Baseline for both is a value of 0 in any case.
The reason for the choice of these two evaluation metrics is that, they consider precision and recall at the same time and they are able to offer more intuitive and onformative measurement of the binary classification performance, which range from -1 to 1. Cohen's Kappa approximate the real effect of the classifier considering random effect (using totalAccuracy - randomAccuracy). MCC delivers more informative measurement compared with other metrics such as recall and accuracy and F1 score in binary classification task, as it considers the size of the positive and negative examples.

# 16.12.2018
### State of the Dataset

First off, we did not collaborate with group zeta on our dataset, as our approach for generating negatives differs vastly from theirs.
Second, as our dataset is still very noisy, we are still working on finding effective mechanisms to tackle that noise. We've implemented a second way to gain the dataset using NLTK, as well as using a custom regular grammar, leading to different datasets, and will use as final dataset the set union of those datasets, which (hopefully) contains less noise than the individual ones.

Other methods still being worked on is a script that checks for entities using the knowledgestore, that extracts information from these. As, however, the knowledgestore-API was really unreliable in the past week (the server was down almost all of the time when we worked on this task), we unfortunately could not finish this.

#### Storing the dataset


We did also re-decide how we're saving our dataset, to make it more efficient: Our csv-file now only contains the minimal necessary information: The article URI, the positions (in terms of indices) of the agent, predicate, and patient, as well as the set this sample belongs to - for example
| URI | Agent | Predicate | Patient | Set |
| --- | --- | --- | --- | --- |
| http://en.wikinews.org/wiki/Zoo_elephants_live_shorter... | [1:10] | [23:27] | [50: 57] | train |

All further information, as for example annotations for entities (as extracted using the knowledgestore, especially the corresponding dbpedia-entries, see *features*) is then created on-demand, when extracting the samples/batches for the training process. As this process is heavily cached, it will not take substantially more time than saving it in the dataset.

Saving our dataset like this will treat different occurences of the same entity as different answer candidates - if the word "Trump" appears two times in the article, one of which will certainly be a false answer, as it is treated as a different candidate than the other instance.

The train/test/validation-split in this file is made such that it is ensured that new context is seen in the validation-set and the test-set.


#### Negative samples

As for negative samples, they are also not saved in the csv, as it bloats the dataset unnecessarily. We implemented a function generating negative samples from positive ones, which simply takes other words of the same article as negatives. This function can then be called on each batch/sample individually. This has the advantage that we can make up negative examples on the spot, as needed, and we don't need to decide for a train/test-split in advance. If we wanted to work with neural networks for example, we could enforce batch balancing by generating 16 positive samples as well as 16 other positive samples that went through the function ```make_negative```, thus becoming negative samples.
Another advantage is that we can modify the ```make_negative``` on demand. Conceivable changes are for example that only other entities (agent/patient) of other positive samples are to be selected to create a negative sample, or that only words of the same part-of-speech shall be selected - But we'll cross that bridge once we get to it.

When doing the inference step of our learner, it is of course the best approach to go through every single word in the text as candidate answer, and simply returning the one yielding the highest value.

The function to make negative samples from positive samples is currently not cached and may slow down the learning process, but should that happen, it will simply be run in a separate pre-fetching process, or implement a cache around it as well.

### Features

`A note on one of Lucas' Questions - I do not understand what is meant by "*do they target positive or negative examples*" - Everything targets positive and negatives examples at once by definition - Something that helps finding positive examples also helps distinguishing it from negative examples.
A feature targeting positive examples without targeting negative examples would be, as far as I understand it, something like 'is a word'. Every positive example is a word, so it does target positive samples. As everything that is not a word is a negative example, it doesn't target negative examples. But, as can be seen here, such features are obviously useless.
A feature that 'targets positive examples' could, for example, be 'is a verb' (if it's looked for relations/predicates). But while this feature targets positive examples (verbs are more likely to be a relation), it (via the law of excluded middle) also targets negative examples, as non-verbs are *less* likely to be a relation.
So, all our features target positive examples as much as they target negative examples - their value should just be statistically different for positive and negative ones.`

*Note that we will often talk about Agent and Predicate as if they are given, and the Patient as if it was searched for, but this is only meant exemplary and with no loss of generality.*

As of now, we are working on different sets of features, which differ vastly in their complexity - a simple one, that hopefully turns out to be a nice proof-of-concept, as well as a complex one that requires much more training, but hopefully will achieve a much higher performance.

#### Simple and Stupid Version

It is a pretty mind-bending question of what to present the learner for our task of question-answering using news articles - for the learner to be as good as possible, what one would want is, of course, some way of incorporating the entire article as input to the learner. In the simple version of it however, we will not do that, but instead only feed it some way of 'surface features', grasping some information about the words of the triple, instead of ones trying to summarize the meaning of the full text.

If, for example, it is asked for the patient of the triple <John Wilkes Booth, shot, Abraham Lincoln> (as in the sentence "Whom did John Wilkes Booth shoot?"), we will provide information about the entity *John-Wilkes-Booth*, the predicate *shoot*, as well as the respective answer candidate (for a positive sample *Abraham Lincoln*, or whatever the answer-candidate will be for a negative sample).

The features we currently extract for these three words are respectively:
* Part-of-speech of the respective words (in the case of multi-word expressions, we will replace the expression with a one-word-token and run another POS-tagger on the resulting sentence to get one unambiguous part-of-speech) *with the hope that agent and patient are generally nouns, whereas verbs tend to be predicates* (this is a categorical value, for example encoded as a one-hot vector)
* Positions of the respective words, *with the hope that the predicate is often in between subject and object, or at least very close* (this is an integer for each of those)
* For all binary combinations of <Subject, Predicate, Object>, the information if they are in the same sentence *with the hope that positive samples tend to be* (this are three booleans in total)

This list can, of course, be arbitrarily extended to include further, in part less 'shallow', features:
* An idea is to not only consider the words themselves, but also a windows of surrounding words (say, two words before and behind the respective word), including their information (POS, position, ...) as well.
* Another idea is to run a dependency-parser[1] on the sentence containing the subject and predicate, saving the start/end positions of the parts of the sentence that are the daughters and parents of the agent/predicate/patient, *with the hope that, for example, the patient is a daughter of the predicate*.
* Feature-Engineering using word2vec: Parsing the sentences containing agent, predicate and patient, and then using the word2vec-representation of subject, verb, and possible objects in this sentence (sums in the case of multi-word-expressions), as well as the word2vec-representations of agent, predicate, and patient *with the hope that, for example, the patient's vector-representation will be similar to that of the object of the sentence containing the predicate*.
* Another feature not yet included is the fact if predicate and agent / predicate and patient match in terms of person, gender, case,  singular/plural, ... - *with the hope that words that do not match grammatically are less likely to be correct answers* (this will be a boolean for each feature).
* Further, annotations from the knowledgestore (and also possibly from WordNet) could be used - some predicates can only be performed by 'agents', so it makes sense to use such information as well. It also makes sense to use super-categories of words as features (again, either using dbpedia (trump -> politician), or using WordNet/FrameNet. 


#### Complex version

As mentioned above, a complex version of input, requiring a lot of training (and probably deep nets) is also imaginable.

One way of doing it is, to create a sentence for every answer-candidate (for example for the triple <John, stole, the cheese>, the sentence would be 'John stole the cheese'), and run a Neural Network trained for the task of textual entailment on this sentence as well as every sentence of the article. Such a neural network returns a likelihood that a given **hypothesis** is entailed by a given **premise**. If the sentence generated by our answer candidate is actually entailed by any of the sentences of the article, it is probably a correct answer to the question (or rather: the candidate answer which has the highest score on any of the sentences in the article is the correct answer).

The advantage of using such textual-entailment-ANNs is, that they encode the entire sentences (by using an LSTM over the individual vector-representations of its words), such that the maximal amount of information is provided, especially when run over every single sentence.  More information on textual entailment is given in the pdf `textual_entailment.pdf`, provided in this repository.

While this version is in theory very promising, it is also vastly complex to implement, and requires much longer training times, such that we will stick to the simple and stupid version of the inputs for now.

### Preprocessing

While the features of both the simple and the complex version can be extracted without much preprocessing, doing so will probably help a lot in the final classification of our learner. One such preprocessing step is **pronoun resolution**. Relatively often in the text, the correct answer to a question is a pronoun. While that may make sense in the article, it is of course an unsatisfactory answer. We're currently in the midst of extracting these pronouns using the knowledgestore-API, and are not sure if it is actually possible (due to the API being down constantly this week). If that does not work, other ways of pronoun-resolution will be used instead. We are, however, not sure if it makes more sense to do the pronoun-resolution in advance, or as a post-processing step (meaning, that if the resulting is a pronoun, we just query it with the knowlegestore-API to get what this pronoun refers to).

When using word2vec as feature, there is always the problem of unseen words - words for that no such word-vector exists. Re-training word2vec on the given corpus is theoretically possible, but increases the amount of trainable parameters by many orders of magnitude (see for example section 3.1.2 of the provided file `textual_entailment.pdf`), such that its not recommended. A possible preprocessing-step may thus be to **replace words for which no wordvectors exist by hypernyms of them**. This is possible by querying dbpedia using the knowledgestore-API - if 'Merkel' would be unknown, it could be replaced by one of its *types* (see the function `get_entity_info` of the file `entities_meta_information.py`): *Politician*. If there are too many *types*, and this cannot be decided, querying WordNet to gain a hypernym of the respective word is also possible.


## References

[1] Joakim Nivre and Mario Scholz. 2004. Deterministic dependency parsing of English text. In Proceedings of the 20th international conference on Computational Linguistics (COLING '04). Association for Computational Linguistics, Stroudsburg, PA, USA, Article 64 . DOI: https://doi.org/10.3115/1220355.1220365
