# ML4NLP - Zeta
Material for the Practical Seminar "Machine Learning for Natural Language Processing" (Institute of Cognitive Science, Osnabrück University, Winter Term 2018/2019).

This is the branch of group Zeta.

## Session 2

### 1. Overall Task
Our goal is to develop a system that is able to take a question phrased in a natural language (English in our case) and find an adequate concise answer to the question in the Wikinews data set. 

### 2. Approach
In order to gain training data, we plan to generate question-answer pairs by utilizing the subject-relation-object structure of RDF(Resource Description Framework) triples. While this limits the amount of questions the system can handle (e.g. “How did Donald Trump become 45th President of the United States?“ would exceed the answering capabilites of our system), causal questions requiring abstractive summarization would massively increase the scope of this project. Moreover, this decision still leaves us with an extensive set of possible questions like "Who owns Facebook?", "What is the relation between Mark Zuckerberg and Facebook?", and "What company does Mark Zuckerberg own?". While the last question seems vague to humans, we chose this example because it corresponds to a straight-forward SPARQL(SPARQL Protocol and RDF Query Language) query.

The initial challenge is translating a question like the ones above into a SPARQL query. 

#### 2.1 Classification Process
We propose to develop classifiers that evaluates a set of triples and assigns probabilities regarding their relevance to a given question. After having determined the most relevant triple, the queried piece of information (subject, relation, or object respectively) is returned as the answer.

This approach is feasibly implementable given our situation since three groups are working on the overall projects and the approach requires three different classifiers for each query scenario (querying the subject, querying the relation, or querying the object of a triple). Each classifier will be developed by one of the three groups.