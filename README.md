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
