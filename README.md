# ML4NLP - Delta
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Delta.

## 4.11.2018

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



## 11.11.2018

### Intro

This week's task is to have the first idea on the labels autogeneration in an elegant manner, which means the autogenerated labels should be representative, low-effort, high-quality, and well documented. As described in last week's documentation, the labels we need are $information-triples$ which can be queried in KnowledgeStore Database. 
Our $information-triples$ include three components, namely two entities and one relation (between two entities). 

### General Workflow 1.0

Goal of this subtask (autogeneration of labels) is to generate $information-triples$, which means two entities and one relation (between entities). This can be further broken down into two subtasks.

1. Entities extraction

one possible way to detect entities is to use pre-existing Name Entity Recognition (NER) database (CoNLL 2003), which will be used to retrieve more meta-information of every sentense in our dataset, in order to better autogenerate $information-triples$. Bi-LSTM could be used in this benchmark to achieve relatively good performance (precision/recall/F score at around 90% [1]). Output of this subtask is the following four items (word, POS tag, chunk tag, NER tag) for each word in the article: the first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. For example, 

* U.N. NNP I-NP I-ORG 
* official NN I-NP O 
* Ekeus NNP I-NP I-PER 
* heads VBZ I-VP O 
* for IN I-PP O 
* Baghdad NNP I-NP I-LOC 

with this we can extract entities for each sentense.

2. Relation extraction

with the class $nwr:RelationMention$ in Knowledgestore API, it is of low cost to extract the relation between two entities. Output of this would be one of the following four different types of relations.

* nwr:TLink - A temporal link, i.e., a mention denoting a temporal relation among two events and/or time expressions. 
* nwr:CLink - A causal link, i.e., a mention denoting a causal relation among two events. 
* nwr:Participation - A mention denoting the participation of an object (e.g., a person) to a certain event, further characterized by the role played by that object. 
* nwr:GLink - A grammatical link among event mentions. 
* nwr:SLink - A structural link, i.e., a mention denoting a structural relation among two events. 


### About NER database CoNLL 2003

This dataset [1] includes 1,393 English and 909 German news articles, which is compatible with our dataset. We can use the english version for free, low cost :D. Our goal in this task is to extract entities for each sentence. Entities are annotated with LOC (location), ORG (organisation), PER (person) and MISC (miscellaneous). For instance, "$John Wilkes Booth shot Abraham Lincoln$" will be annotated with "$John[PER] Wilkes[PER] Booth[PER] shot[O] Abraham[PER] Lincoln[PER]$". You might notice that $John Wilkes Booth$ will be annotated with three entity tags, rather than one single tag. How to overcome this problem needs more investigation.

Some ANN-based models are able to achieve around 90% precision, recall and F score in this benchmark, such as [2]. The choice of architecture still remains to be explored. But B-LSTM could be a good choice, with GloVe or GoogleNews Word2vec. 

### desiderata

One biggest advantage of our approach is that it can generate entities for each sentense, which can be further combined with Knowledgestore method to extract the relation between entities. 

- large: for each $possible$ sentence in every article the information-triples will be extracted. To this extent, we are reaching the maximal possibility to get a large data.
- low-effort: absolutely. it's for free and there is an open-source implementation which achieves the current state-of-the-art performance (https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL).
- representative: we limit our scope into tackling answerable questions. Under this constraint, our approach of extracting entities and relation seems to be representative.
- high-quality: Because the NER task will introduce some unavoidable loss, so the quality of labels can not reach to optimal level. However, 90% F score seems to be good enough to keep a trade-off between quality and effort. After extracting information-triples, some manual checks are also expected.
- well-documented: absolutely. With task of NER we will receive a part-of-speech (POS) tag, a syntactic chunk tag and a named entity tag in word-level. Further combined with Knowledgestore API we can retrieve more meta-data. 

### Limitation

1. step-by-step fashion of breaking down the problem will accumulate loss.
2. Training for NER requires huge computation power and take. (the most power machine we are able to access is a 12G Titan X in IKW-Grid, which is still runnable with reasonable parameters of the ANN model.)
3. Probably it will take some time to tune the model in order to achieve better performance.


## References

[1] Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition." Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4. Association for Computational Linguistics, 2003.

[2] Chiu, Jason PC, and Eric Nichols. "Named entity recognition with bidirectional LSTM-CNNs." arXiv preprint arXiv:1511.08308 (2015).