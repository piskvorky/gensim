from __future__ import print_function

import collections
import logging
from itertools import chain, product
import math
import random

_logger = logging.getLogger(__name__)

EvaluationConfig = collections.namedtuple('EvaluationConfig',
                                          ['num_samples', 'sample_size'])

FORMAT_STRINGS = {
    'default': """Filename   : {name}
Num samples: {samplesize_count}
Sample size: {samplesize_avg}
F-score    : {fscore_avg:.3}
Precision  : {precision_avg:.3}
Recall     : {recall_avg:.3}""",
    'table': "{name:10} {precision_avg:6.3} {recall_avg:6.3} {fscore_avg:6.3}",
    'latex': "{name} & {precision_avg:.3} &"
             " {recall_avg:.3} & {fscore_avg:.3} \\\\"}


def _sample(compound_list, size, seed):
    """Create a specific size sample from the compound list using a specific
    seed"""
    return random.Random(seed).sample(compound_list, size)


class MorfessorEvaluationResult(object):
    """A MorfessorEvaluationResult is returned by a MorfessorEvaluation
    object. It's purpose is to store the evaluation data and provide nice
    formatting options.

    Each MorfessorEvaluationResult contains the data of 1 evaluation
    (which can have multiple samples).

    """

    print_functions = {'avg': lambda x: sum(x) / len(x),
                       'min': min,
                       'max': max,
                       'values': list,
                       'count': len}
    #TODO add maybe std as a print function?

    def __init__(self, meta_data=None):
        self.meta_data = meta_data

        self.precision = []
        self.recall = []
        self.fscore = []
        self.samplesize = []

        self._cache = None

    def __getitem__(self, item):
        """Provide dict style interface for all values (standard values and
        metadata)"""
        if self._cache is None:
            self._fill_cache()

        return self._cache[item]

    def add_data_point(self, precision, recall, f_score, sample_size):
        """Method used by MorfessorEvaluation to add the results of a single
        sample to the object"""
        self.precision.append(precision)
        self.recall.append(recall)
        self.fscore.append(f_score)
        self.samplesize.append(sample_size)

        #clear cache
        self._cache = None

    def __str__(self):
        """Method for default visualization"""
        return self.format(FORMAT_STRINGS['default'])

    def _fill_cache(self):
        """ Pre calculate all variable / function combinations and put them in
        cache"""
        self._cache = {'{}_{}'.format(val, func_name): func(getattr(self, val))
                       for val in ('precision', 'recall', 'fscore',
                                   'samplesize')
                       for func_name, func in self.print_functions.items()}
        self._cache.update(self.meta_data)

    def _get_cache(self):
        """ Fill the cache (if necessary) and return it"""
        if self._cache is None:
            self._fill_cache()
        return self._cache

    def format(self, format_string):
        """ Format this object. The format string can contain all variables,
        e.g. fscore_avg, precision_values or any item from metadata"""
        return format_string.format(**self._get_cache())


class MorfessorEvaluation(object):
    """ Do the evaluation of one model, on one testset. The basic procedure is
    to create, in a stable manner, a number of samples and evaluate them
    independently. The stable selection of samples makes it possible to use
    the resulting values for Pair-wise statistical significance testing.

    reference_annotations is a standard annotation dictionary:
    {compound => ([annoation1],.. ) }
    """
    def __init__(self, reference_annotations):
        self.reference = {}

        for compound, analyses in reference_annotations.items():
            self.reference[compound] = list(
                tuple(self._segmentation_indices(a)) for a in analyses)

        self._samples = {}

    def _create_samples(self, configuration=EvaluationConfig(10, 1000)):
        """Create, in a stable manner, n testsets of size x as defined in
        test_configuration
        """

        #TODO: What is a reasonable limit to warn about a too small testset?
        if len(self.reference) < (configuration.num_samples *
                                  configuration.sample_size):
            _logger.warn("The test set is too small for this sample size")

        compound_list = sorted(self.reference.keys())
        self._samples[configuration] = [
            _sample(compound_list, configuration.sample_size, i) for i in
            range(configuration.num_samples)]

    def get_samples(self, configuration=EvaluationConfig(10, 1000)):
        """Get a list of samples. A sample is a list of compounds.

        This method is stable, so each time it is called with a specific
        test_set and configuration it will return the same samples. Also this
        method caches the samples in the _samples variable.

        """
        if not configuration in self._samples:
            self._create_samples(configuration)
        return self._samples[configuration]

    def _evaluate(self, prediction):
        """Helper method to get the precision and recall of 1 sample"""
        def calc_prop_distance(ref, pred):
            if len(ref) == 0:
                return 1.0
            diff = len(set(ref) - set(pred))
            return (len(ref) - diff) / float(len(ref))

        wordlist = sorted(set(prediction.keys()) & set(self.reference.keys()))

        recall_sum = 0.0
        precis_sum = 0.0

        for word in wordlist:
            if len(word) < 2:
                continue

            recall_sum += max(calc_prop_distance(r, p)
                              for p, r in product(prediction[word],
                                                  self.reference[word]))

            precis_sum += max(calc_prop_distance(p, r)
                              for p, r in product(prediction[word],
                                                  self.reference[word]))

        precision = precis_sum / len(wordlist)
        recall = recall_sum / len(wordlist)
        f_score = 2.0 / (1.0 / precision + 1.0 / recall)

        return precision, recall, f_score, len(wordlist)

    @staticmethod
    def _segmentation_indices(annotation):
        """Method to transform a annotation into a tuple of split indices"""
        cur_len = 0
        for a in annotation[:-1]:
            cur_len += len(a)
            yield cur_len

    def evaluate_model(self, model, configuration=EvaluationConfig(10, 1000),
                       meta_data=None):
        """Get the prediction of the test samples from the model and do the
        evaluation

        The meta_data object has preferably at least the key 'name'.

        """
        if meta_data is None:
            meta_data = {'name': 'UNKNOWN'}

        mer = MorfessorEvaluationResult(meta_data)

        for i, sample in enumerate(self.get_samples(configuration)):
            _logger.debug("Evaluating sample {}".format(i))
            prediction = {}
            for compound in sample:
                prediction[compound] = [tuple(self._segmentation_indices(
                    model.viterbi_segment(compound)[0]))]

            mer.add_data_point(*self._evaluate(prediction))

        return mer

    def evaluate_segmentation(self, segmentation,
                              configuration=EvaluationConfig(10, 1000),
                              meta_data=None):
        """Method for evaluating an existing segmentation"""

        def merge_constructions(constructions):
            compound = constructions[0]
            for i in range(1, len(constructions)):
                compound = compound + constructions[i]
            return compound

        segmentation = {merge_constructions(x[1]):
                        [tuple(self._segmentation_indices(x[1]))]
                        for x in segmentation}

        if meta_data is None:
            meta_data = {'name': 'UNKNOWN'}

        mer = MorfessorEvaluationResult(meta_data)

        for i, sample in enumerate(self.get_samples(configuration)):
            _logger.debug("Evaluating sample {}".format(i))

            prediction = {k: v for k, v in segmentation.items() if k in sample}
            mer.add_data_point(*self._evaluate(prediction))

        return mer


class WilcoxonSignedRank(object):
    """Class for doing statistical signficance testing with the Wilcoxon
    Signed-Rank test

    It implements the Pratt method for handling zero-differences and
    applies a 0.5 continuity correction for the z-statistic.

    """

    @staticmethod
    def _wilcoxon(d, method='pratt', correction=True):
        if method not in ('wilcox', 'pratt'):
            raise ValueError
        if method == 'wilcox':
            d = list(filter(lambda a: a != 0, d))

        count = len(d)

        ranks = WilcoxonSignedRank._rankdata([abs(v) for v in d])
        rank_sum_pos = sum(r for r, v in zip(ranks, d) if v > 0)
        rank_sum_neg = sum(r for r, v in zip(ranks, d) if v < 0)

        test = min(rank_sum_neg, rank_sum_pos)

        mean = count * (count + 1) * 0.25
        stdev = (count*(count + 1) * (2 * count + 1))
        # compensate for duplicate ranks
        no_zero_ranks = [r for i, r in enumerate(ranks) if d[i] != 0]
        stdev -= 0.5 * sum(x * (x*x-1) for x in
                           collections.Counter(no_zero_ranks).values())

        stdev = math.sqrt(stdev / 24.0)

        if correction:
            correction = +0.5 if test > mean else -0.5
        else:
            correction = 0
        z = (test - mean - correction) / stdev

        return 2 * WilcoxonSignedRank._norm_cum_pdf(abs(z))

    @staticmethod
    def _rankdata(d):
        od = collections.Counter()
        for v in d:
            od[v] += 1

        rank_dict = {}
        cur_rank = 1
        for val, count in sorted(od.items(), key=lambda x: x[0]):
            rank_dict[val] = (cur_rank + (cur_rank + count - 1)) / 2
            cur_rank += count

        return [rank_dict[v] for v in d]

    @staticmethod
    def _norm_cum_pdf(z):
        """Pure python implementation of the normal cumulative pdf function"""
        return 0.5 - 0.5 * math.erf(z / math.sqrt(2))

    def significance_test(self, evaluations, val_property='fscore_values',
                          name_property='name'):
        """Takes a set of evaluations (which should have the same
        test-configuration) and calculates the p-value for the Wilcoxon signed
        rank test

        Returns a dictionary with (name1,name2) keys and p-values as values.
        """
        results = {r[name_property]: r[val_property] for r in evaluations}
        if any(len(x) < 10 for x in results.values()):
            _logger.error("Too small number of samples for the Wilcoxon test")
            return {}
        p = {}
        for r1, r2 in product(results.keys(), results.keys()):
            p[(r1, r2)] = self._wilcoxon([v1-v2
                                          for v1, v2 in zip(results[r1],
                                                            results[r2])])

        return p

    @staticmethod
    def print_table(results):
        """Nicely format a results table as returned by significance_test"""
        names = sorted(set(r[0] for r in results.keys()))

        col_width = max(max(len(n) for n in names), 5)

        for h in chain([""], names):
            print('{:{width}}'.format(h, width=col_width), end='|')
        print()

        for name in names:
            print('{:{width}}'.format(name, width=col_width), end='|')

            for name2 in names:
                print('{:{width}.5}'.format(results[(name, name2)],
                                            width=col_width), end='|')
            print()
