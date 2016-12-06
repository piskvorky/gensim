
# coding: utf-8
"""
Word2Vec_FastText_Comparison.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

# # Comparison of FastText and Word2Vec

# Facebook Research open sourced a great project recently - [fastText](https://github.com/facebookresearch/fastText), a fast (no surprise) and effective method to learn word representations and perform text classification. I was curious about comparing these embeddings to other commonly used embeddings, so word2vec seemed like the obvious choice, especially considering fastText embeddings are an extension of word2vec.
#
# I've used gensim to train the word2vec models, and the analogical reasoning task (described in Section 4.1 of [[2]](https://arxiv.org/pdf/1301.3781v3.pdf)) for comparing the word2vec and fastText models. I've compared embeddings trained using the skipgram architecture.

# # Download data

# In[1]:

import nltk
nltk.download('brown')
# Only the brown corpus is needed in case you don't have it.

# Generate brown corpus text file
with open('brown_corp.txt', 'w+') as f:
    for word in nltk.corpus.brown.words():
        f.write('{word} '.format(word=word))

# Make sure you set FT_HOME to your fastText directory root
FT_HOME = 'fastText/'
# download the text8 corpus (a 100 MB sample of cleaned wikipedia text)
import os.path
if not os.path.isfile('text8'):
    # get_ipython().system(u'wget -c http://mattmahoney.net/dc/text8.zip')
    # get_ipython().system(u'unzip text8.zip')
    os.system(u'wget -c http://mattmahoney.net/dc/text8.zip')
    os.system(u'unzip text8.zip')

# download and preprocess the text9 corpus
if not os.path.isfile('text9'):
  # get_ipython().system(u'wget -c http://mattmahoney.net/dc/enwik9.zip')
  # get_ipython().system(u'unzip enwik9.zip')
  # get_ipython().system(u'perl {FT_HOME}wikifil.pl enwik9 > text9')
  os.system(u'wget -c http://mattmahoney.net/dc/enwik9.zip')
  os.system(u'unzip enwik9.zip')
  os.system(u'perl {FT_HOME}wikifil.pl enwik9 > text9')


# # Train models

# For training the models yourself, you'll need to have both [Gensim](https://github.com/RaRe-Technologies/gensim) and [FastText](https://github.com/facebookresearch/fastText) set up on your machine.

# In[2]:

MODELS_DIR = 'models/'
# get_ipython().system(u'mkdir -p {MODELS_DIR}')
# os.system(u'mkdir -p {MODELS_DIR}')

lr = 0.05
dim = 100
ws = 5
epoch = 5
minCount = 5
neg = 5
loss = 'ns'
t = 1e-4

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

# Same values as used for fastText training above
params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    'min_count': minCount,
    'sample': t,
    'sg': 1,
    'hs': 0,
    'negative': neg
}

def train_models(corpus_file, output_name):
    output_file = '{0:s}_ft'.format(output_name)
    if not os.path.isfile(os.path.join(MODELS_DIR, '{0:s}.vec'.format(output_file))):
        print('Training fasttext on {0:s} corpus..'.format(corpus_file))
        # get_ipython().magic(u'time !{FT_HOME}fasttext skipgram -input {corpus_file} -output {MODELS_DIR+output_file}  -lr {lr} -dim {dim} -ws {ws} -epoch {epoch} -minCount {minCount} -neg {neg} -loss {loss} -t {t}')
    else:
        print('\nUsing existing model file {0:s}.vec'.format(output_file))

    output_file = '{0:s}_ft_no_ng'.format(output_name)
    if not os.path.isfile(os.path.join(MODELS_DIR, '{0:s}.vec'.format(output_file))):
        print('\nTraining fasttext on {0:s} corpus (without char n-grams)..'.format(corpus_file))
        # get_ipython().magic(u'time !{FT_HOME}fasttext skipgram -input {corpus_file} -output {MODELS_DIR+output_file}  -lr {lr} -dim {dim} -ws {ws} -epoch {epoch} -minCount {minCount} -neg {neg} -loss {loss} -t {t} -maxn 0')
    else:
        print('\nUsing existing model file {0:s}.vec'.format(output_file))

    output_file = '{0:s}_gs'.format(output_name)
    if not os.path.isfile(os.path.join(MODELS_DIR, '{0:s}.vec'.format(output_file))):
        print('\nTraining word2vec on {0:s} corpus..'.format(corpus_file))

        # Text8Corpus class for reading space-separated words file
        # get_ipython().magic(u'time gs_model = Word2Vec(Text8Corpus(corpus_file), **params); gs_model')
        # Direct local variable lookup doesn't work properly with magic statements (%time)
        locals()['gs_model'].save_word2vec_format(os.path.join(MODELS_DIR, '{0:s}.vec'.format(output_file)))
        print('\nSaved gensim model as {0:s}.vec'.format(output_file))
    else:
        print('\nUsing existing model file {0:s}.vec'.format(output_file))

evaluation_data = {}
train_models('brown_corp.txt', 'brown')


# In[3]:

train_models(corpus_file='text8', output_name='text8')


# In[4]:

train_models(corpus_file='text9', output_name='text9')


# # Comparisons

# In[15]:

# download the file questions-words.txt to be used for comparing word embeddings
# get_ipython().system(u'wget https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt')
os.system(u'wget https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt')


# Once you have downloaded or trained the models and downloaded `questions-words.txt`, you're ready to run the comparison.

# In[14]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Training times in seconds
evaluation_data['brown'] = [(18, 54.3, 32.5)]
evaluation_data['text8'] = [(402, 942, 496)]
evaluation_data['text9'] = [(3218, 6589, 3550)]

def print_accuracy(model, questions_file):
    print('Evaluating...\n')
    acc = model.accuracy(questions_file)

    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100*float(sem_correct)/sem_total
    print('\nSemantic: {0:d}/{1:d}, Accuracy: {2:.2f}%'.format(sem_correct, sem_total, sem_acc))

    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc)-1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5,len(acc)-1))
    syn_acc = 100*float(syn_correct)/syn_total
    print('Syntactic: {0:d}/{1:d}, Accuracy: {2:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return (sem_acc, syn_acc)

word_analogies_file = 'questions-words.txt'
accuracies = []
print('\nLoading Gensim embeddings')
brown_gs = Word2Vec.load_word2vec_format(MODELS_DIR + 'brown_gs.vec')
print('Accuracy for Word2Vec:')
accuracies.append(print_accuracy(brown_gs, word_analogies_file))

print('\nLoading FastText embeddings')
brown_ft = Word2Vec.load_word2vec_format(MODELS_DIR + 'brown_ft.vec')
print('Accuracy for FastText (with n-grams):')
accuracies.append(print_accuracy(brown_ft, word_analogies_file))


# Word2Vec embeddings seem to be slightly better than fastText embeddings at the semantic tasks, while the fastText embeddings do significantly better on the syntactic analogies. Makes sense, since fastText embeddings are trained for understanding morphological nuances, and most of the syntactic analogies are morphology based.
#
# Let me explain that better.
#
# According to the paper [[1]](https://arxiv.org/abs/1607.04606), embeddings for words are represented by the sum of their n-gram embeddings. This is meant to be useful for morphologically rich languages - so theoretically, the embedding for `apparently` would include information from both character n-grams `apparent` and `ly` (as well as other n-grams), and the n-grams would combine in a simple, linear manner. This is very similar to what most of our syntactic tasks look like.
#
# Example analogy:
#
# `amazing amazingly calm calmly`
#
# This analogy is marked correct if:
#
# `embedding(amazing)` - `embedding(amazingly)` = `embedding(calm)` - `embedding(calmly)`
#
# Both these subtractions would result in a very similar set of remaining ngrams.
# No surprise the fastText embeddings do extremely well on this.
#
# Let's do a small test to validate this hypothesis - fastText differs from word2vec only in that it uses char n-gram embeddings as well as the actual word embedding in the scoring function to calculate scores and then likelihoods for each word, given a context word. In case char n-gram embeddings are not present, this reduces (atleast theoretically) to the original word2vec model. This can be implemented by setting 0 for the max length of char n-grams for fastText.
#

# In[15]:

print('Loading FastText embeddings')
brown_ft_no_ng = Word2Vec.load_word2vec_format(MODELS_DIR + 'brown_ft_no_ng.vec')
print('Accuracy for FastText (without n-grams):')
accuracies.append(print_accuracy(brown_ft_no_ng, word_analogies_file))
evaluation_data['brown'] += [[acc[0] for acc in accuracies], [acc[1] for acc in accuracies]]


# A-ha! The results for FastText with no n-grams and Word2Vec look a lot more similar (as they should) - the differences could easily result from differences in implementation between fastText and Gensim, and randomization. Especially telling is that the semantic accuracy for FastText has improved slightly after removing n-grams, while the syntactic accuracy has taken a giant dive. Our hypothesis that the char n-grams result in better performance on syntactic analogies seems fair. It also seems possible that char n-grams hurt semantic accuracy a little. However, the brown corpus is too small to be able to draw any definite conclusions - the accuracies seem to vary significantly over different runs.

# Let's try with a larger corpus now - text8 (collection of wiki articles). I'm also curious about the impact on semantic accuracy - for models trained on the brown corpus, the difference in the semantic accuracy and the accuracy values themselves are too small to be conclusive. Hopefully a larger corpus helps, and the text8 corpus likely has a lot more information about capitals, currencies, cities etc, which should be relevant to the semantic tasks.

# In[16]:

accuracies = []
print('Loading Gensim embeddings')
text8_gs = Word2Vec.load_word2vec_format(MODELS_DIR + 'text8_gs.vec')
print('Accuracy for word2vec:')
accuracies.append(print_accuracy(text8_gs, word_analogies_file))

print('Loading FastText embeddings (with n-grams)')
text8_ft = Word2Vec.load_word2vec_format(MODELS_DIR + 'text8_ft.vec')
print('Accuracy for FastText (with n-grams):')
accuracies.append(print_accuracy(text8_ft, word_analogies_file))

print('Loading FastText embeddings')
text8_ft_no_ng = Word2Vec.load_word2vec_format(MODELS_DIR + 'text8_ft_no_ng.vec')
print('Accuracy for FastText (without n-grams):')
accuracies.append(print_accuracy(text8_ft_no_ng, word_analogies_file))

evaluation_data['text8'] += [[acc[0] for acc in accuracies], [acc[1] for acc in accuracies]]


# With the text8 corpus, we observe a similar pattern. Semantic accuracy falls by a small but significant amount when n-grams are included in FastText, while FastText with n-grams performs far better on the syntactic analogies. FastText without n-grams are largely similar to Word2Vec.
#
# My hypothesis for semantic accuracy being lower for the FastText-with-ngrams model is that most of the words in the semantic analogies are standalone words and are unrelated to their morphemes (eg: father, mother, France, Paris), hence inclusion of the char n-grams into the scoring function actually makes the embeddings worse.
#
# This trend is observed in the original paper too where the performance of embeddings with n-grams is worse on semantic tasks than both word2vec cbow and skipgram models.
#
# Let's do a quick comparison on an even larger corpus - text9

# In[17]:

accuracies = []
print('Loading Gensim embeddings')
text9_gs = Word2Vec.load_word2vec_format(MODELS_DIR + 'text9_gs.vec')
print('Accuracy for word2vec:')
accuracies.append(print_accuracy(text9_gs, word_analogies_file))

print('Loading FastText embeddings (with n-grams)')
text9_ft = Word2Vec.load_word2vec_format(MODELS_DIR + 'text9_ft.vec')
print('Accuracy for FastText (with n-grams):')
accuracies.append(print_accuracy(text9_ft, word_analogies_file))

print('Loading FastText embeddings')
text9_ft_no_ng = Word2Vec.load_word2vec_format(MODELS_DIR + 'text9_ft_no_ng.vec')
print('Accuracy for FastText (without n-grams):')
accuracies.append(print_accuracy(text9_ft_no_ng, word_analogies_file))

evaluation_data['text9'] += [[acc[0] for acc in accuracies], [acc[1] for acc in accuracies]]


# In[23]:

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

def plot(ax, data, corpus_name='brown'):
    width = 0.25
    pos = [(i, i + width, i + 2*width) for i in range(len(data))]
    colors = ['#EE3224', '#F78F1E', '#FFC222']
    acc_ax = ax.twinx()
    # Training time
    ax.bar(pos[0],
            data[0],
            width,
            alpha=0.5,
            color=colors
            )
    # Semantic accuracy
    acc_ax.bar(pos[1],
            data[1],
            width,
            alpha=0.5,
            color=colors
            )

    # Syntactic accuracy
    acc_ax.bar(pos[2],
            data[2],
            width,
            alpha=0.5,
            color=colors
            )

    ax.set_ylabel('Training time (s)')
    acc_ax.set_ylabel('Accuracy (%)')
    ax.set_title(corpus_name)

    acc_ax.set_xticks([p[0] + 1.5 * width for p in pos])
    acc_ax.set_xticklabels(['Training Time', 'Semantic Accuracy', 'Syntactic Accuracy'])

    # Proxy plots for adding legend correctly
    proxies = [ax.bar([0], [0], width=0, color=c, alpha=0.5)[0] for c in colors]
    models = ('Gensim', 'FastText', 'FastText (no-ngrams)')
    ax.legend((proxies), models, loc='upper left')

    ax.set_xlim(pos[0][0]-width, pos[-1][0]+width*4)
    ax.set_ylim([0, max(data[0])*1.1] )
    acc_ax.set_ylim([0, max(data[1] + data[2])*1.1] )

    plt.grid()

# Plotting the bars
fig = plt.figure(figsize=(10,15))
for corpus, subplot in zip(sorted(evaluation_data.keys()), [311, 312, 313]):
    ax = fig.add_subplot(subplot)
    plot(ax, evaluation_data[corpus], corpus)

plt.show()


# The results from text9 seem to confirm our hypotheses so far. Briefly summarising the main points -
#
# 1. FastText models with n-grams do significantly better on syntactic tasks, because of the syntactic questions being related to morphology of the words
# 2. Both Gensim word2vec and the fastText model with no n-grams do slightly better on the semantic tasks, presumably because words from the semantic questions are standalone words and unrelated to their char n-grams
# 3. In general, the performance of the models seems to get closer with the increasing corpus size. However, this might possibly be due to the size of the model staying constant at 100, and a larger model size for large corpora might result in higher performance gains.
# 4. The semantic accuracy for all models increases significantly with the increase in corpus size.
# 5. However, the increase in syntactic accuracy from the increase in corpus size for the n-gram FastText model is lower (in both relative and absolute terms). This could possibly indicate that advantages gained by incorporating morphological information could be less significant in case of larger corpus sizes (the corpuses used in the original paper seem to indicate this too)
# 6. Training times for gensim are slightly lower than the fastText no-ngram model, and significantly lower than the n-gram variant. This is quite impressive considering fastText is implemented in C++ and Gensim in Python (with calls to low-level BLAS routines for much of the heavy lifting). You could read [this post](http://rare-technologies.com/word2vec-in-python-part-two-optimizing/) for more details regarding word2vec optimisation in Gensim. Note that these times include importing any dependencies and serializing the models to disk, and not just the training times.

# # Conclusions

# These preliminary results seem to indicate fastText embeddings are significantly better than word2vec at encoding syntactic information. This is expected, since most syntactic analogies are morphology based, and the char n-gram approach of fastText takes such information into account. The original word2vec model seems to perform better on semantic tasks, since words in semantic analogies are unrelated to their char n-grams, and the added information from irrelevant char n-grams worsens the embeddings. It'd be interesting to see how transferable these embeddings are for different kinds of tasks by comparing their performance in a downstream supervised task.

# # References

# [1] [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)
#
# [2] [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781v3.pdf)
