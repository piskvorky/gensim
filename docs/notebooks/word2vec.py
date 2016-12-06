
# coding: utf-8
"""
word2vec.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # Word2Vec Tutorial
# This tutorial follows a [blog post](http://rare-technologies.com/word2vec-tutorial/) written by the creator of gensim.

# ## Preparing the Input
# Starting from the beginning, gensim’s `word2vec` expects a sequence of sentences as its input. Each sentence a list of words (utf8 strings):

# In[1]:

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)


# Keeping the input as a Python built-in list is convenient, but can use up a lot of RAM when the input is large.
#
# Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything in RAM: we can provide one sentence, process it, forget it, load another sentence…
#
# For example, if our input is strewn across several files on disk, with one sentence per line, then instead of loading everything into an in-memory list, we can process the input file by file, line by line:

# In[3]:

# create some toy data to use with the following example
import smart_open, os

if not os.path.exists('./data/'):
    os.makedirs('./data/')

filenames = ['./data/f1.txt', './data/f2.txt']

for i, fname in enumerate(filenames):
    with smart_open.smart_open(fname, 'w') as fout:
        for line in sentences[i]:
            fout.write(line + '\n')


# In[4]:

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


# In[5]:

sentences = MySentences('./data/') # a memory-friendly iterator
print(list(sentences))


# In[6]:

# generate the Word2Vec model
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model)
print(model.vocab)


# Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract named entities… All of this can be done inside the `MySentences` iterator and `word2vec` doesn’t need to know. All that is required is that the input yields one sentence (list of utf8 words) after another.
#
# **Note to advanced users:** calling `Word2Vec(sentences)` will run two passes over the sentences iterator.
#   1. The first pass collects words and their frequencies to build an internal dictionary tree structure.
#   2. The second pass trains the neural model.
#
# These two passes can also be initiated manually, in case your input stream is non-repeatable (you can only afford one pass), and you’re able to initialize the vocabulary some other way:

# In[7]:

# build the same model, making the 2 steps explicit
new_model = gensim.models.Word2Vec(min_count=1)  # an empty model, no training
new_model.build_vocab(sentences)                 # can be a non-repeatable, 1-pass generator
new_model.train(sentences)                       # can be a non-repeatable, 1-pass generator
print(new_model)
print(model.vocab)


# ## More data would be nice
# For the following examples, we'll use the Lee Corpus (which you already have if you've installed gensim):

# In[8]:

# Set file names for train and test data
test_data_dir = '{0}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'


# In[9]:

class MyText(object):
    def __iter__(self):
        for line in open(lee_train_file):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()

sentences = MyText()

print(sentences)


# ## Training
# `Word2Vec` accepts several parameters that affect both training speed and quality.
#
# One of them is for pruning the internal dictionary. Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage. In addition, there’s not enough data to make any meaningful training on those words, so it’s best to ignore them:

# In[10]:

# default value of min_count=5
model = gensim.models.Word2Vec(sentences, min_count=10)


# In[11]:

# default value of size=100
model = gensim.models.Word2Vec(sentences, size=200)


# Bigger size values require more training data, but can lead to better (more accurate) models. Reasonable values are in the tens to hundreds.
#
# The last of the major parameters (full list [here](http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)) is for training parallelization, to speed up training:

# In[12]:

# default value of workers=3 (tutorial says 1...)
model = gensim.models.Word2Vec(sentences, workers=4)


# The `workers` parameter only has an effect if you have [Cython](http://cython.org/) installed. Without Cython, you’ll only be able to use one core because of the [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) (and `word2vec` training will be [miserably slow](http://rare-technologies.com/word2vec-in-python-part-two-optimizing/)).

# ## Memory
# At its core, `word2vec` model parameters are stored as matrices (NumPy arrays). Each array is **#vocabulary** (controlled by min_count parameter) times **#size** (size parameter) of floats (single precision aka 4 bytes).
#
# Three such matrices are held in RAM (work is underway to reduce that number to two, or even one). So if your input contains 100,000 unique words, and you asked for layer `size=200`, the model will require approx. `100,000*200*4*3 bytes = ~229MB`.
#
# There’s a little extra memory needed for storing the vocabulary tree (100,000 words would take a few megabytes), but unless your words are extremely loooong strings, memory footprint will be dominated by the three matrices above.

# ## Evaluating
# `Word2Vec` training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application.
#
# Google have released their testing set of about 20,000 syntactic and semantic test examples, following the “A is to B as C is to D” task. You can download a zip file [here](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip), and unzip it, to get the `questions-words.txt` file used below.

# Gensim support the same evaluation set, in exactly the same format:

# In[13]:

try:
    model.accuracy('questions-words.txt')
except FileNotFoundError:
    raise ValueError("SKIP: please download the questions-word.txt file.")


# This `accuracy` takes an
# [optional parameter](http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.accuracy) `restrict_vocab`
# which limits which test examples are to be considered.
#
# Once again, **good performance on this test set doesn’t mean word2vec will work well in your application, or vice versa**. It’s always best to evaluate directly on your intended task.

# ## Storing and loading models
# You can store/load models using the standard gensim methods:

# In[14]:

model.save('/tmp/mymodel')
new_model = gensim.models.Word2Vec.load('/tmp/mymodel')


# which uses pickle internally, optionally `mmap`‘ing the model’s internal large NumPy matrices into virtual memory directly from disk files, for inter-process memory sharing.
#
# In addition, you can load models created by the original C tool, both using its text and binary formats:
#
#     model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
#     # using gzipped/bz2 input works too, no need to unzip:
#     model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)

# ## Online training / Resuming training
# Advanced users can load a model and continue training it with more sentences:

# In[15]:

model = gensim.models.Word2Vec.load('/tmp/mymodel')
more_sentences = ['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue',
                  'training', 'it', 'with', 'more', 'sentences']
model.train(more_sentences)


# You may need to tweak the `total_words` parameter to `train()`, depending on what learning rate decay you want to simulate.
#
# Note that it’s not possible to resume training with models generated by the C tool, `load_word2vec_format()`. You can still use them for querying/similarity, but information vital for training (the vocab tree) is missing there.
#
# ## Using the model
# `Word2Vec` supports several word similarity tasks out of the box:

# In[16]:

model.most_similar(positive=['human', 'crime'], negative=['party'], topn=1)


# In[17]:

model.doesnt_match("input is lunch he sentence cat".split())


# In[18]:

print(model.similarity('human', 'party'))
print(model.similarity('tree', 'murder'))


# If you need the raw output vectors in your application, you can access these either on a word-by-word basis:

# In[19]:

model['tree']  # raw NumPy vector of a word


# …or en-masse as a 2D NumPy matrix from `model.syn0`.
#
# ## Outro
# There is a **Bonus App** on the original [blog post](http://rare-technologies.com/word2vec-tutorial/), which runs `word2vec` on the Google News dataset, of **about 100 billion words**.
#
# Full `word2vec` API docs [here](http://radimrehurek.com/gensim/models/word2vec.html); get [gensim](http://radimrehurek.com/gensim/) here. Original C toolkit and `word2vec` papers by Google [here](https://code.google.com/archive/p/word2vec/).
