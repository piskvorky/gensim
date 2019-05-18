r"""

.. _fasttext_tutorial:

How to Use Gensim as a Wrapper for FastText
===========================================

This tutorial is about using `fastText <https://github.com/facebookresearch/fastText>`_ model in Gensim.

"""

###############################################################################
#
# There are two ways you can use fastText in Gensim - Gensim's native implementation of fastText and Gensim wrapper for fastText's original C++ code.
# This howto explains the latter.
# For the former, see :ref:`fasttext_tutorial`.


from gensim.models.wrappers.fasttext import FastText as FT_wrapper


###############################################################################
#
# You will need fastText setup locally to be able to train models.
# See `installation instructions for fastText <https://github.com/facebookresearch/fastText/#requirements>`_ if you don't have fastText installed already.
#

# Set FastText home to the path to the FastText executable
ft_home = '/home/misha/src/fastText-0.1.0/fasttext'

# train the model
from gensim.test.utils import datapath

corpus_file = datapath('lee_background.cor')
model_wrapper = FT_wrapper.train(ft_home, corpus_file)
print(model_wrapper)

###############################################################################
#
# Perform common operations with the model
# For a more detailed introduction to FastText, :ref:`fasttext_tutorial`.
# The native and wrapped model behave identically, and differ in implementation only.
#

print('night' in model_wrapper.wv.vocab)

###############################################################################
#
print('nights' in model_wrapper.wv.vocab)

###############################################################################
#
print(model_wrapper['night'])

###############################################################################
#
print(model_wrapper['nights'])

###############################################################################
# Tests if word present in vocab
#
print("word" in model_wrapper.wv.vocab)

###############################################################################
# Tests if vector present for word
#
print("word" in model_wrapper)

###############################################################################
# saving a model trained via fastText wrapper
#
model_wrapper.save('saved_model_wrapper')
loaded_model = FT_wrapper.load('saved_model_wrapper')
print(loaded_model)
