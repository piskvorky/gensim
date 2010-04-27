.. _tut3:

Similarity Queries
===========================

.. warning::
   TODO

To print proximity of this query document against every one of the nine original 
documents, in the same 2-D LSI space:

>>> index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and "index" it
>>> sims = index[lsi[doc_bow]] # perform similarity query against the corpus
>>> print list(enumerate(sims))
[(0, 0.99809301), (1, 0.93748635), (2, 0.99844527), (3, 0.9865886), (4, 0.90755945), 
(5, -0.12416792), (6, -0.1063926), (7, -0.098794639), (8, 0.05004178)]

The thing to note here is that documents no. 2 (``"The EPS user interface management system"``)
and 4 (``"Relation of user perceived response time to error measurement"``) would never be returned by
a standard boolean fulltext search, because they do not share any common words with ``"Human 
computer interaction"``. However, after applying LSI, we can observe that both of 
them received quite high similarity scores, which corresponds better to our intuition of
them sharing a "computer-related" topic with the query. In fact, this is the reason 
why we apply transformations and do topic modeling in the first place.

