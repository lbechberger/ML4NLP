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
As discussed before, we wan to make use of the data contained within the KnowledgeStore database. We desire a large amount of data to enable more complciated machine learning approaches, and we want data clean of formating/labeling issues, that could break our pipeline. As such, we decided to autogenerate the data from the DB if possible. Furthermore, we decided that we want to focus on the Relation component of the triples, i.e. the answer the question "What is the relation between Mr Bigshot and the BigCompany." -> "Mr Bigshot owns the BigCompany/The BigCompany is owned by Mr Bigshot."
Researching the database and the entries in the database revealed that the entries that we seek, so called "Mentions"  usually have an attribute called "RelationMention". (Link to the wiki here:) However,  SPARQL seems to be unable to direcly filter for this attribute. Currently we explore a combination of a primary SPARQL query requesting the combination of Mentions, their article and their attributes, and a secundary python script that futher filterse these based on the attribute "RelationMention". 
From these filtered queries we desire to generate the test and trainings data sets. However, as mention in "Choice of Input and Target" we are not quite sure if we also want to add realworld knowledge about the entries in the article via further SPARQL queries. 
 
 ###  Pros and Cons
 As mentioned, we wanted a large and syntactically clean data set, which is possible with autogeneration. However, we are aware of the limitations. We cannot surpass the semnatic quality of the KnowledgeStore, nor can we generate information not yet contained within it, possibly an issue if we want to apply the algorythm to articles outside of the DB. For the articles contained within, however, we are quite sure to be capable of generating representative QA pairs, as these articles are related in their content and style. 
Object to label: Mentions, Instances where we can generate a complete triple.
NWR: RelationMention usually has the required information, but SPARQL cant filter for it. 

### Further steps required 
We stated that our goal was to answer natural language QA pairs. Currently we only talked about the generation of incomplete and complete triplets. However, as we limited the style of our questions greatly, the translation to natural language can be done by simply "filling in the blanks". "Who is the mother fo Eddadottir" --> (?, is mother of, Eddadottir) --> (Edda,is mother of, Eddadottir) --> "Edda is the mother of Eddadottir." 

#### Notes
Here are notes for the group. If we forget to delete this, please ignore them.
We can generate the required lables from the triplet information. Turning that into NL requires fillign in blanks

Size we can guarentee due to the size of the DB
Representative is given for the articles, as we select information from the same articles in order to answers about these articles
The level of effort will be reasonable: Currently we only see a requirement for a simple SPARQL query, combined with an additional filter to select RelationMention Entries
 Qualitywise, it will be perfectly clean as it is autogenerated, but semantically, it can only be as good as the knowledge store.
Concerning the documentation, well, you are reading it.

We do it this way because it was the best way could think of. (We wanted large amounts of data, we have a large, high-quality database)
