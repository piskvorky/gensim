
# coding: utf-8
"""
plot_summarization_tutorial.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# <h1>Tutorial: automatic summarization using Gensim</h1>
#
# This module automatically summarizes the given text, by extracting one or more important sentences from the text. In a similar way, it can also extract keywords. This tutorial will teach you to use this summarization module via some examples. First, we will try a small example, then we will try two larger ones, and then we will review the performance of the summarizer in terms of speed.
#
# This summarizer is based on the "TextRank" algorithm, from an [article](http://web.eecs.umich.edu/%7Emihalcea/papers/mihalcea.emnlp04.pdf) by Mihalcea et al. This algorithm was later improved upon by Barrios et al. in another [article](https://raw.githubusercontent.com/summanlp/docs/master/articulo/articulo-en.pdf), by introducing something called a "BM25 ranking function".
#
# This tutorial assumes that you are familiar with Python and have [installed Gensim](http://radimrehurek.com/gensim/install.html).
#
# <b>Note</b>: Gensim's summarization only works for English for now, because the text is pre-processed so that stopwords are removed and the words are stemmed, and these processes are language-dependent.
#
#
# <h2>Small example</h2>
#
# First of all, we import the function "summarize".

# In[1]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.summarization import summarize


# We will try summarizing a small toy example; later we will use a larger piece of text. In reality, the text is too small, but it suffices as an illustrative example.

# In[2]:

text = "Thomas A. Anderson is a man living two lives. By day he is an " +     "average computer programmer and by night a hacker known as " +     "Neo. Neo has always questioned his reality, but the truth is " +     "far beyond his imagination. Neo finds himself targeted by the " +     "police when he is contacted by Morpheus, a legendary computer " +     "hacker branded a terrorist by the government. Morpheus awakens " +     "Neo to the real world, a ravaged wasteland where most of " +     "humanity have been captured by a race of machines that live " +     "off of the humans' body heat and electrochemical energy and " +     "who imprison their minds within an artificial reality known as " +     "the Matrix. As a rebel against the machines, Neo must return to " +     "the Matrix and confront the agents: super-powerful computer " +     "programs devoted to snuffing out Neo and the entire human " +     "rebellion. "

print ('Input text:')
print (text)


# To summarize this text, we pass the <b>raw string data</b> as input to the function "summarize", and it will return a summary.
#
# Note: make sure that the string does not contain any newlines where the line breaks in a sentence. A sentence with a newline in it (i.e. a carriage return, "\n") will be treated as two sentences.

# In[3]:

print ('Summary:')
print (summarize(text))


# Use the "split" option if you want a list of strings instead of a single string.

# In[4]:

print (summarize(text, split=True))


# You can adjust how much text the summarizer outputs via the "ratio" parameter or the "word_count" parameter. Using the "ratio" parameter, you specify what fraction of sentences in the original text should be returned as output. Below we specify that we want 50% of the original text (the default is 20%).

# In[5]:

print ('Summary:')
print (summarize(text, ratio=0.5))


# Using the "word_count" parameter, we specify the maximum amount of words we want in the summary. Below we have specified that we want no more than 50 words.

# In[6]:

print ('Summary:')
print (summarize(text, word_count=50))


# As mentioned earlier, this module also supports <b>keyword</b> extraction. Keyword extraction works in the same way as summary generation (i.e. sentence extraction), in that the algorithm tries to find words that are important or seem representative of the entire text. They keywords are not always single words; in the case of multi-word keywords, they are typically all nouns.

# In[7]:

from gensim.summarization import keywords

print ('Keywords:')
print (keywords(text))


# <h2>Larger example</h2>
#
# Let us try an example with a larger piece of text. We will be using a synopsis of the movie "The Matrix", which we have taken from [this](http://www.imdb.com/title/tt0133093/synopsis?ref_=ttpl_pl_syn) IMDb page.
#
# In the code below, we read the text file directly from a web-page using "requests". Then we produce a summary and some keywords.

# In[ ]:

import requests

text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text

print ('Summary:')
print (summarize(text, ratio=0.01))

print ('\nKeywords:')
print (keywords(text, ratio=0.01))


# If you know this movie, you see that this summary is actually quite good. We also see that some of the most important characters (Neo, Morpheus, Trinity) were extracted as keywords.
#
# <h2>Another example</h2>
#
# Let's try an example similar to the one above. This time, we will use the [IMDb synopsis](http://www.imdb.com/title/tt0118715/synopsis?ref_=tt_stry_pl) of "The Big Lebowski".
#
# Again, we download the text and produce a summary and some keywords.

# In[ ]:

import requests

text = requests.get('http://rare-technologies.com/the_big_lebowski_synopsis.txt').text

print ('Summary:')
print (summarize(text, ratio=0.01))

print ('\nKeywords:')
print (keywords(text, ratio=0.01))


# This time around, the summary is not of high quality, as it does not tell us much about the movie. In a way, this might not be the algorithms fault, rather this text simply doesn't contain one or two sentences that capture the essence of the text as in "The Matrix" synopsis.
#
# The keywords, however, managed to find some of the main characters.
#
# <h2>Performance</h2>
#
# We will test how the speed of the summarizer scales with the size of the dataset. These tests were run on an Intel Core i5 4210U CPU @ 1.70 GHz x 4 processor. Note that the summarizer does <i>not</i> support multithreading (parallel processing).
#
# The tests were run on the book "Honest Abe" by Alonzo Rothschild. Download the book in plain-text <a href="http://www.gutenberg.org/ebooks/49679">here</a>.
#
# In the <b>plot below</b>, we see the running times together with the sizes of the datasets. To create datasets of different sizes, we have simply taken prefixes of text; in other words we take the first <i>n</i> characters of the book. The algorithm seems to be <b>quadratic in time</b>, so one needs to be careful before plugging a large dataset into the summarizer.
#
# <figure>
# <img src="http://rare-technologies.com/summarization_tutorial_plot.png">
#     <figcaption></figcaption>
# </figure>
#
# <h3>Text-content dependent running times</h3>
#
# The running time is not only dependent on the size of the dataset. For example, summarizing "The Matrix" synopsis (about 36,000 characters) takes about 3.1 seconds, while summarizing 35,000 characters of this book takes about 8.5 seconds. So the former is <i>more than twice as fast</i>.
#
# One reason for this difference in running times is the data structure that is used. The algorithm represents the data using a graph, where vertices (nodes) are sentences, and then constructs weighted edges between the vertices that represent how the sentences relate to each other. This means that every piece of text will have a different graph, thus making the running times different. The size of this data structure is <i>quadratic in the worst case</i> (the worst case is when each vertex has an edge to every other vertex).
#
# Another possible reason for the difference in running times is that the problems converge at different rates, meaning that the error drops slower for some datasets than for others.
#
