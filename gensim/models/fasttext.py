#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Prakhar Pratyush (er.prakhar2b@gmail.com)

"""
TO-DO : description of FastText and the API
"""

"""
void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w + c]);  // n-grams here
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}
"""

def train_cbow():
    # borrowed from word2vec, see how much is overlapping and refactor later

    #TO-DO : get n-grams ahd continue rest same as word2vec


def train_skipgram():

@staticmethod
    def ft_hash(string):
        """
        Reproduces [hashing trick](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        used in fastText.

        """
        # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
        old_settings = np.seterr(all='ignore')
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h



class FastText(Word2Vec):
	# TO-DO : check if sentences can be None here like word2vec ?
	def __init__(self, model='cbow', sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            word_ngrams=1, loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000):

        # TO-Discuss : these param names vs fb fastText param names ?

		"""
		Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (utf-8 encoded strings) that will be used for training.

        `model` defines the training algorithm. By default, cbow is used. Accepted values are
        'cbow', 'skipgram', (later 'supervised').  ------- decide if sg=0 default cbow is better approach

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate.

        `min_count` = ignore all words with total occurrences lower than this.

        `word_ngram` = max length of word ngram

        `loss` = defines training objective. Allowed values are `hs` (hierarchical softmax),
        `ns` (negative sampling) and `softmax`. Defaults to `ns`

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `negative` = the value for negative specifies how many "noise words" should be drawn
        (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        Only relevant when `loss` is set to `ns`

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `min_n` = min length of char ngrams to be used for training word representations. Default is 3.

        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        lesser than `min_n` to avoid char ngrams being used. Default is 6.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.



		"""
		self.initialize_word_vectors()

		self.model = model
		self.vector_size = size
        self.alpha = float(alpha)
        self.window = int(window)
        self.min_count = min_count
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.sample = sample
        self.negative = negative
        self.iter = iter
        self.bucket = bucket

        self.min_n = min_n
        self.max_n = max_n

        # if (wordNgrams <= 1 && maxn == 0) {
        #    bucket = 0;
        # }

        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

            # TO-DO : do we need 

            self.train()


	def initialize_word_vectors():
		# approach from wrapper

        self.word_vectors = FastTextKeyedVectors
        # TO-DO : backward-compatibility with self.wv (under discussion)

    def train(self, sentences):

        # epochs in word2vec ??

        # input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
        # input_->uniform(1.0 / args_->dim);

        # create a matrix for n-grams here

        # output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
        # This is probably for generating .vec file

        # Start with thread = 0 i.e, trainThread(0) in fasttext.cc

        if self.model == 'cbow':
            train_cbow(self, sentences, self.alpha)
        elif self.model == 'skipgram':
            train_skipgram(self, sentences, self.alpha)

