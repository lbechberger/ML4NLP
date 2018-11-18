# ML4NLP - Zeta

Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).


This is the branch of group Zeta.

- - - -

## Week 2


### 1. Overall Task

Our goal is to develop a system that is able to take a question phrased in a natural language (English in our case) and find an adequate concise answer to the question in the Wikinews data set.

### 2. Approach

In order to gain training data, we plan to generate question-answer pairs by utilizing the subject-relation-object structure of RDF(Resource Description Framework) triples. While this limits the amount of questions the system can handle (e.g. “How did Donald Trump become 45th President of the United States?“ would exceed the answering capabilites of our system), causal questions requiring abstractive summarization would massively increase the scope of this project. Moreover, this decision still leaves us with an extensive set of possible questions like "Who owns Facebook?", "What is the relation between Mark Zuckerberg and Facebook?", and "What company does Mark Zuckerberg own?". While the last question seems vague to humans, we chose this example because it corresponds to a straight-forward SPARQL(SPARQL Protocol and RDF Query Language) query.


The initial challenge is translating a question like the ones above into a SPARQL query.


#### 2.1 Classification Process

We propose to develop classifiers that evaluates a set of triples and assigns probabilities regarding their relevance to a given question. After having determined the most relevant triple, the queried piece of information (subject, relation, or object respectively) is returned as the answer.


This approach is feasibly implementable given our situation since three groups are working on the overall projects and the approach requires three different classifiers for each query scenario (querying the subject, querying the relation, or querying the object of a triple). Each classifier will be developed by one of the three groups.

- - - -

## Week 3

For our approach, we need a set of triples that is as large as possible, is quick and easy to obtain, and corresponds to the labels in the KnowledgeStore database. We have decided to use an autogeneration approach to construct a dataset that fulfills these requirements.

### 3. Data Set Generation

As mentioned above, RDF triples abide an subject-relation-object structure. Given the fact that we have reduced the scope of the task to specific questions regarding named entities (e.g. „Who is the father of Malia Obama?“), we know that all possible subjects and objects for the RDF triples we can use are named entities. Luckily, the NewsReader Ontology (https://knowledgestore.fbk.eu/ontologies/newsreader.html) that is also used in the KnowledgeStore database has a class to label named entities called EntityMention. By extracting all instances of the type nwr:EntityMention, we have assembled a complete list of all the potential subjects/objects of triples.

After having identified all possible subjects/objects, relations between all possible subject/object pairs need to be evaluated. The NewsReader Ontology has a suitable class for this as well, namely nwr:RelationMention. This class denotes the relation between two distinct entites and potentially conveys additional information regarding the type of relation through its subclasses nwr:TLink (temporal link), nwr:CLink (causal link), nwr:Participation, nwr:GLink (grammatical link), and nwr:SLink (structural link). This enables us to query for relations in all possible subject/object combinations and use the result of each query to generate a triple.

#### 3.1 Potential Challenges

The entities that we can choose by this aproach are limited to the ones that have been labelled by instances of EntityMention, RelationMention etc. and we have yet to evaluate how thoroughly the news articles have been labeled in this regard.

#### 3.2 Properties of the Data Set

#### 3.2.1 Size

Fairly large since we search for entities in the sentences in an article using ObjectMention and this corresponds to a relatively large number of instances. Moreover, we will query for relations between all possible subject/object-combinations.


#### 3.2.2 Representativeness

It is representative since all the triples (based on their instances) are from the same articles in the database.

#### 3.2.3 Low Effort

Relatively low effort to auto generate question answer pairs.

#### 3.2.4 Quality

We expect it to be decent at best since they are auto generated. Also the quality is inherently influenced by the correctness of the labels present in the data itself for instances of EntityMention, RelationMention etc.

#### 3.2.5 Documentation

We hope to provide a comprehensive documentation.

- - - -

## Week 4

As discussed in the previous week, as the first step, our data contains RDF triples generated from wikinews articles. To achieve this, we have decided to use tools provided by the Natural Language Toolkit platform (http://www.nltk.org/) for named-entity recognition (NER). Using the entities extracted in this way, we can proceed to relation extraction. Coincidentally, NLTK also offers tools for this exact purpose as well. Therefore, the workflow for the first part of dataset generation will be to generate a list of strings containing *n* (actual quantity not yet decided upon) wikinews articles. The following workflow will look thus:
1. split into sentences
2. sentences will be POS-tagged
3. NER will be applied to each sentence
4. Relation extraction between fully tagged (both POS and NER) sentences will be applied
The product of this approach will be sets of subject-relation-object items that will be saved in a simple text-file for now. POS- and NER-tags will also be included in the text-file. 
The second step involves generating potential questions for each part of a generated triple. Given time constraints, we lack a concrete implementation of this just now but the plan is to use common question phrases adequate for the kind of inquiry. We are currently considering to also add morphological analysis in the first step of our data generation to make question generation even more adequate (taking into consideration singular vs. plural, for example).

> Mark Zuckerberg [Pers] - founded - Facebook [Org]
> > Who is [subject]?\
> What is [object]?\
> How are [subject] and [object] related?

The fact that our previous processing has provided us with syntactic information regarding the constituents of a triple, the automatically generated questions can be grammatically adequate given the word they inquire about.

Once we have generated questions that could query for information in a given triple, these triple-question sets will be stored in an XML file containing the triple and then potential questions for each constituent of the triple.