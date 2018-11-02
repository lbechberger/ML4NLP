# ML4NLP - Epsilon
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Epsilon.

## Week 29.10-04.11

### Stated goal
We aim to create a system that is capable of answering questions concerning a single document. I.e. When asked "Who owns the Company BigCompay?" after reading a text about this company, it should be able to answer "Mrs Bigshot". For this we make use of the  KnowledgeStore Database (link: https://knowledgestore.fbk.eu/), and the articles contained within.

### Approach
We quickly realised that a complete QA system capable of parsing  difficulty questions ("Whose fault is the decline of the Euro, according to Mrs Moneybags?") would be beyond our scope. Inspired by the triplets that make up entries in SPARQL, we decided to limit the questions to the style of (Who has relation x to y?), (What is the relation between z and y?) and (Z has relation x to who/what?).
This enables us to make use of these triplets to autogenerate large amount of questions, making heavy duty machine learning feasible. 
Furthermore this decision enabled the splitting of the task into 3 variants, one for each style of question. This is especially suited for our approach, as we consist of three groups, working in parallel. 

### Input/Target 
We decided that the target of the predicitons ought to be the completed triple of the (Subjet,Relation,Object) form. Most likely all potiential targets in an article are given a probability to be the proper target and the most likely will be classified as the answer.
Concerning the input, our descision has not yet been finalised. While we are sure that the article about which the question is posed ought to be part of the it, there are further open questions. Should the complete KnowledgeStore database be made available to the algorithm in some form, to represent world knowledge? Can we input the complete article, or should we use only the sentences containing two of the three triplets? The sentence +/- one more? Continued discussion is required here.  

### Model Choices
So far, possible models have only briefly been discussed. The only certainty is the we will make use of a classification algorithm, due to outside constrains. 


