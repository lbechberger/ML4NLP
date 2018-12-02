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

- - - -

## Week 5

### 5. Data Set Generation (Advanced)
The process of preliminary data set generation is drawing to a close. The only remaining steps are relation extraction, question generation, and storing the generated information in the most usable format.
#### 5.1 Relation Extraction
Once named entities have been identified in a text, we try to extract relations between specified types of named entity. One way of approaching this task is to initially look for all triples of the form (X, α, Y), where X and Y are named entities of the required types, and α is the string of words that intervenes between X and Y. We can then use regular expressions to pull out just those instances of α that express the relation that we are looking for. The following example searches for strings that contain the word in. The special regular expression (?!\b.+ing\b) is a negative lookahead assertion that allows us to disregard strings such as “success in supervising the transition of”, where in is followed by a gerund.

We can then use the function rtuple to print the form (Sub, filler, Obj), where Sub and Obj are pairs of Named Entity mentions, and filler is the string of words occurring between Sub and Obj (with no intervening NEs). For example:
[ORG: 'Christian Democrats'] ', the leading political forces in' [LOC: 'Italy']

Searching for the keyword *in* works reasonably well, though it will also retrieve false positives such as [ORG: House Transportation Committee], secured the most money in the [LOC: New York]; there is unlikely to be simple string-based method of excluding filler strings such as this.
We can also define the pattern (α relation) as a disjunction of roles that a PERSON can occupy in an ORGANIZATION like (director| economist| editor etc). In addition to this, the “conll2002” and “ace” corpus contains not just named entity annotation but also part-of-speech tags and thus we can include POS tags in the query pattern.


#### 5.2 Question Generation
In this week's session, we received information that the system does not *have to* handle proper natural language questions, simple question formats (hereafter referred to as 'artificial questions') would also be acceptable. However, we decided to give it a try anyway. We are therefore working on both approaches, working on two methods to generate natural language questions and artificial questions. 
##### 5.2.1 Natural Language Questions
The two biggest issues with natural language questions (NLQ) is the sheer variety of possible phrasings as well as the amount of information needed about the syntactic nature of lexical items and the context they are used in. The fact that RDF triples are based on a structure reminiscing of natural language sentences (subject-verb-object structure), we can already determine the case of the question by the position of the information in a triple. You can find an example for why cases matter in the example under section 5.3. The gender of subject and object, however, are not inherently encoded in the information and should therefore be gathered in another way. The naive approach of querying for subjects and objects simply with the term "who" would result in unnatural NLQs like "Whom does Mark Zuckerberg own?" instead of "What does Mark Zuckerberg own?". We therefore preserved morphological information for every word in a given text. In order to achieve this, we changed our POS-tagging solution to the RDRPOSTagger (https://github.com/datquocnguyen/RDRPOSTagger) which is also able to derive morphological information. The accuracy of its POS-tagging for the English language reached up to 96.49% [1]. We have not yet evaluated its accuracy ourselves, however. Moreover, no claim about its accuracy for morphological tagging was made as of yet.
##### 5.2.2 Artificial Questions
The generation of artificial question (AQ) sets is significantly less complicated than the generation of NLQs. AQs are of the format (Barack_Obama, father_of, ?), (?, father_of, Malia_Obama), or (Barack_Obama, ?, Malia_Obama). We do not need any syntactic information for any items in the triple, instead we just generate strings of these three formats for every triple.

#### 5.3 Storing the Information
Given the circumstance that we have not yet decided on a concrete implementation of the classifier and the labelling format we want to apply to our data, we decided to store the information we generate in an XML file. With this as a foundation, we can later automatically transform the stored data into the format we want to train our classifier with.

    <qa-set>
        <triple> "Barack Obama" - "father to" - "Malia Obama" </triple>
        <subject_question> "Who is the father of Malia Obama?" </subject_question>
        <object_question> "Whom is Barack Obama father to?" </object_question>
        <object_question> "Whose father is Barack Obama?" </object_question>
        ....
    </qa-set>

It should be noted that a full qa-set should consist of one triple and a set of subject questions, a set of relation questions, and a set of object questions. This formatting style allows us to store all the information needed to later add labels in any shape we might need them. In addition to this, this dataset could in theory be used to train classifiers for subject questions, relation questions, and object questions. Should a later project be to develop a multi-class classifier, a single training data set for that classifier could also be generated from this dataset. We therefore found this formatting to be the most resilient and versatile, giving us more freedom for later design choices.

- - - -

## Week 6

### 6.1 Relation Extraction
In continuation to last week's efforts we are currently using 3 patterns for relation extraction - 'OF' between PER (Person) and GPE (Geopolitical Entities), 'IN' between PER and LOC (Location) and ROLES between PER and ORG (Organization). In our code, ROLES is a a combination of 22 user defined positions that could be held by a person in an organization (for example analyst, governor, writer). Until final execution of the code, we are continuing our efforts to find more patterns as well as looking for the above patterns between other combinations of the named entities.

### 6.2 Negatives Generation
Our previously designed implementation to generate data has so far omitted the need of negatives, i.e. question-answer pairs that do not match. If we do want to create a dataset to train a classifier on, however, we do need those as well. Our approach for this is to take generated question-answer pairs and to replace one of the subjects of the triple it was generated from with another entity mentioned in the text the triple was extracted from. As an example, take the a look at the qa-set example above (the Barack Obama one). It could have been extracted from a text also mentioning Joseph Biden in the sentence after the one that resulted in the given triple. We are currently working on altering the *recognize_ne(sentence)* function of **file_generation.py** to save named entities (including the relative position of their mentioning in the source text) from a given article in a separate list. This list is then going to be used in **question_generation.py** within a new function called *negatives_generation(qa_pair)* which generates two negatives by first replacing the subject of the triple with another entity from the text and saving this as a question-answer set and then by replacing the object of the original triple with another entity from the text and saving that as a question-answer set. Calling the function *negatives_generation(qa_pair)* with the qa-pair being the one from the example above (see 5.3) will add two new subtrees like these to the XML file:
   
    <neg-qa-set>
        <triple> "Joe Biden" - "father to" - "Malia Obama" </triple>
        <subject_question> "Who is the father of Malia Obama?" </subject_question>
        <object_question> "Whom is Barack Obama father to?" </object_question>
        <object_question> "Whose father is Barack Obama?" </object_question>
        ....
    </neg-qa-set>

    
    <neg-qa-set>
        <triple> "Barack Obama" - "father to" - "the White House" </triple>
        <subject_question> "Who is the father of Malia Obama?" </subject_question>
        <object_question> "Whom is Barack Obama father to?" </object_question>
        <object_question> "Whose father is Barack Obama?" </object_question>
        ....
    </neg-qa-set>

#### 6.2.1 Additional Data Collection
In addition to the new way of generating negatives, we have come to the realization that we need to enrich our dataset with additional data regarding the individual tokens of a triple. We will therefore attach a new subtree to every qa-set (and neg-qa-set) in the XML file including information regarding POS of the word in question, type of named entity, and relative position of the word in a text. 

#### 6.2.2 Concerns regarding Noise in the Data Set
Given this newly added amount of data for each qa-pair and this comparably naive approach to generating negatives, we need to point out a flaw in automatically generated datasets which will most likely be present in our dataset as well. We are referring to errors in the automatically labelled data which can occur at many steps. We are using the following (potentially fallible) natural language processing tools in our data generation pipeline: 
- NLTK POS tagging
- NLTK NER
- [Bhaskar's relation extraction thingy]
- a naive NER-matching solution for generation of negatives

Along every step of the process, these tools may make mistakes. NER, for example, turned "Barack Obama" into "Barack" as a person and "Obama" into a seperate entity labelled as an organization during one of our performance evaluations. These mismatches quickly accumulate given the size of our desired dataset and will make for a noisy data set, a circumstance we will have to deal with at a later point in development. We have evaluated our pipeline and do not see any way to reduce the chance of generating noise in our data any further.

# References
[1] Dat Quoc Nguyen, Dai Quoc Nguyen, Dang Duc Pham and Son Bao Pham. RDRPOSTagger: A Ripple Down Rules-based Part-Of-Speech Tagger. In Proceedings of the Demonstrations at the 14th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2014, pp. 17-20, 2014.