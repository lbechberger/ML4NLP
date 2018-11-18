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
