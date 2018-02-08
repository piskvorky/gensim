import collections
import heapq
import logging
import math
import numbers
import random
import re

from .utils import _progress, _is_string
from .exception import MorfessorException, SegmentOnlyModelException

_logger = logging.getLogger(__name__)


def _constructions_to_str(constructions):
    """Return a readable string for a list of constructions."""
    if _is_string(constructions[0]):
        # Constructions are strings
        return ' + '.join(constructions)
    else:
        # Constructions are not strings (should be tuples of strings)
        return ' + '.join(map(lambda x: ' '.join(x), constructions))


# rcount = root count (from corpus)
# count = total count of the node
# splitloc = integer or tuple. Location(s) of the possible splits for virtual
#            constructions; empty tuple or 0 if real construction
ConstrNode = collections.namedtuple('ConstrNode',
                                    ['rcount', 'count', 'splitloc'])


class BaselineModel(object):
    """Morfessor Baseline model class.

    Implements training of and segmenting with a Morfessor model. The model
    is complete agnostic to whether it is used with lists of strings (finding
    phrases in sentences) or strings of characters (finding morphs in words).

    """

    penalty = -9999.9

    def __init__(self, forcesplit_list=None, corpusweight=None,
                 use_skips=False, nosplit_re=None):
        """Initialize a new model instance.

        Arguments:
            forcesplit_list: force segmentations on the characters in
                               the given list
            corpusweight: weight for the corpus cost
            use_skips: randomly skip frequently occurring constructions
                         to speed up training
            nosplit_re: regular expression string for preventing splitting
                          in certain contexts

        """

        # In analyses for each construction a ConstrNode is stored. All
        # training data has a rcount (real count) > 0. All real morphemes
        # have no split locations.
        self._analyses = {}

        # Flag to indicate the model is only useful for segmentation
        self._segment_only = False

        # Cost variables
        self._lexicon_coding = LexiconEncoding()
        self._corpus_coding = CorpusEncoding(self._lexicon_coding)
        self._annot_coding = None

        #Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

        # Configuration variables
        self._use_skips = use_skips  # Random skips for frequent constructions
        self._supervised = False

        # Counter for random skipping
        self._counter = collections.Counter()
        if forcesplit_list is None:
            self.forcesplit_list = []
        else:
            self.forcesplit_list = forcesplit_list
        if nosplit_re is None:
            self.nosplit_re = None
        else:
            self.nosplit_re = re.compile(nosplit_re, re.UNICODE)

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

        self._corpus_weight_updater.update(self, 0)

    def _check_segment_only(self):
        if self._segment_only:
            raise SegmentOnlyModelException()

    @property
    def tokens(self):
        """Return the number of construction tokens."""
        return self._corpus_coding.tokens

    @property
    def types(self):
        """Return the number of construction types."""
        return self._corpus_coding.types - 1  # do not include boundary

    def _add_compound(self, compound, c):
        """Add compound with count c to data."""
        self._corpus_coding.boundaries += c
        self._modify_construction_count(compound, c)
        oldrc = self._analyses[compound].rcount
        self._analyses[compound] = \
            self._analyses[compound]._replace(rcount=oldrc + c)

    def _remove(self, construction):
        """Remove construction from model."""
        rcount, count, splitloc = self._analyses[construction]
        self._modify_construction_count(construction, -count)
        return rcount, count

    def _random_split(self, compound, threshold):
        """Return a random split for compound.

        Arguments:
            compound: compound to split
            threshold: probability of splitting at each position

        """
        splitloc = tuple(i for i in range(1, len(compound))
                         if random.random() < threshold)
        return self._splitloc_to_segmentation(compound, splitloc)

    def _set_compound_analysis(self, compound, parts, ptype='rbranch'):
        """Set analysis of compound to according to given segmentation.

        Arguments:
            compound: compound to split
            parts: desired constructions of the compound
            ptype: type of the parse tree to use

        If ptype is 'rbranch', the analysis is stored internally as a
        right-branching tree. If ptype is 'flat', the analysis is stored
        directly to the compound's node.

        """
        if len(parts) == 1:
            rcount, count = self._remove(compound)
            self._analyses[compound] = ConstrNode(rcount, 0, tuple())
            self._modify_construction_count(compound, count)
        elif ptype == 'flat':
            rcount, count = self._remove(compound)
            splitloc = self._segmentation_to_splitloc(parts)
            self._analyses[compound] = ConstrNode(rcount, count, splitloc)
            for constr in parts:
                self._modify_construction_count(constr, count)
        elif ptype == 'rbranch':
            construction = compound
            for p in range(len(parts)):
                rcount, count = self._remove(construction)
                prefix = parts[p]
                if p == len(parts) - 1:
                    self._analyses[construction] = ConstrNode(rcount, 0,
                                                              0)
                    self._modify_construction_count(construction, count)
                else:
                    suffix = self._join_constructions(parts[p + 1:])
                    self._analyses[construction] = ConstrNode(rcount, count,
                                                              len(prefix))
                    self._modify_construction_count(prefix, count)
                    self._modify_construction_count(suffix, count)
                    construction = suffix
        else:
            raise MorfessorException("Unknown parse type '%s'" % ptype)

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations.

        For semi-supervised models, select the most likely alternative
        analyses included in the annotations of the compounds.

        """
        if not self._supervised:
            return

        # Collect constructions from the most probable segmentations
        # and add missing compounds also to the unannotated data
        constructions = collections.Counter()
        for compound, alternatives in self.annotations.items():
            if not compound in self._analyses:
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for m in analysis:
                constructions[m] += self._analyses[compound].rcount

        # Apply the selected constructions in annotated corpus coding
        self._annot_coding.set_constructions(constructions)
        for m, f in constructions.items():
            count = 0
            if m in self._analyses and not self._analyses[m].splitloc:
                count = self._analyses[m].count
            self._annot_coding.set_count(m, count)

    def _best_analysis(self, choices):
        """Select the best analysis out of the given choices."""
        bestcost = None
        bestanalysis = None
        for analysis in choices:
            cost = 0.0
            for m in analysis:
                if m in self._analyses and not self._analyses[m].splitloc:
                    cost += (math.log(self._corpus_coding.tokens) -
                             math.log(self._analyses[m].count))
                else:
                    cost -= self.penalty  # penalty is negative
            if bestcost is None or cost < bestcost:
                bestcost = cost
                bestanalysis = analysis
        return bestanalysis, bestcost

    def _force_split(self, compound):
        """Return forced split of the compound."""
        if len(self.forcesplit_list) == 0:
            return [compound]
        clen = len(compound)
        j = 0
        parts = []
        for i in range(1, clen):
            if compound[i] in self.forcesplit_list:
                if len(compound[j:i]) > 0:
                    parts.append(compound[j:i])
                parts.append(compound[i:i + 1])
                j = i + 1
        if j < clen:
            parts.append(compound[j:])
        return [p for p in parts if len(p) > 0]

    def _test_skip(self, construction):
        """Return true if construction should be skipped."""
        if construction in self._counter:
            t = self._counter[construction]
            if random.random() > 1.0 / max(1, t):
                return True
        self._counter[construction] += 1
        return False

    def _viterbi_optimize(self, compound, addcount=0, maxlen=30):
        """Optimize segmentation of the compound using the Viterbi algorithm.

        Arguments:
          compound: compound to optimize
          addcount: constant for additive smoothing of Viterbi probs
          maxlen: maximum length for a construction

        Returns list of segments.

        """
        clen = len(compound)
        if clen == 1:  # Single atom
            return [compound]
        if self._use_skips and self._test_skip(compound):
            return self.segment(compound)
        # Collect forced subsegments
        parts = self._force_split(compound)
        # Use Viterbi algorithm to optimize the subsegments
        constructions = []
        for part in parts:
            constructions += self.viterbi_segment(part, addcount=addcount,
                                                  maxlen=maxlen)[0]
        self._set_compound_analysis(compound, constructions, ptype='flat')
        return constructions

    def _recursive_optimize(self, compound):
        """Optimize segmentation of the compound using recursive splitting.

        Returns list of segments.

        """
        if len(compound) == 1:  # Single atom
            return [compound]
        if self._use_skips and self._test_skip(compound):
            return self.segment(compound)
        # Collect forced subsegments
        parts = self._force_split(compound)
        if len(parts) == 1:
            # just one part
            return self._recursive_split(compound)
        self._set_compound_analysis(compound, parts)
        # Use recursive algorithm to optimize the subsegments
        constructions = []
        for part in parts:
            constructions += self._recursive_split(part)
        return constructions

    def _recursive_split(self, construction):
        """Optimize segmentation of the construction by recursive splitting.

        Returns list of segments.

        """
        if len(construction) == 1:  # Single atom
            return [construction]
        if self._use_skips and self._test_skip(construction):
            return self.segment(construction)
        rcount, count = self._remove(construction)

        # Check all binary splits and no split
        self._modify_construction_count(construction, count)
        mincost = self.get_cost()
        self._modify_construction_count(construction, -count)
        splitloc = 0
        for i in range(1, len(construction)):
            if (self.nosplit_re and
                    self.nosplit_re.match(construction[(i - 1):(i + 1)])):
                continue
            prefix = construction[:i]
            suffix = construction[i:]
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            cost = self.get_cost()
            self._modify_construction_count(prefix, -count)
            self._modify_construction_count(suffix, -count)
            if cost <= mincost:
                mincost = cost
                splitloc = i

        if splitloc:
            # Virtual construction
            self._analyses[construction] = ConstrNode(rcount, count,
                                                      splitloc)
            prefix = construction[:splitloc]
            suffix = construction[splitloc:]
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            lp = self._recursive_split(prefix)
            if suffix != prefix:
                return lp + self._recursive_split(suffix)
            else:
                return lp + lp
        else:
            # Real construction
            self._analyses[construction] = ConstrNode(rcount, 0, tuple())
            self._modify_construction_count(construction, count)
            return [construction]

    def _modify_construction_count(self, construction, dcount):
        """Modify the count of construction by dcount.

        For virtual constructions, recurses to child nodes in the
        tree. For real constructions, adds/removes construction
        to/from the lexicon whenever necessary.

        """
        if construction in self._analyses:
            rcount, count, splitloc = self._analyses[construction]
        else:
            rcount, count, splitloc = 0, 0, 0
        newcount = count + dcount
        if newcount == 0:
            del self._analyses[construction]
        else:
            self._analyses[construction] = ConstrNode(rcount, newcount,
                                                      splitloc)
        if splitloc:
            # Virtual construction
            children = self._splitloc_to_segmentation(construction, splitloc)
            for child in children:
                self._modify_construction_count(child, dcount)
        else:
            # Real construction
            self._corpus_coding.update_count(construction, count, newcount)
            if self._supervised:
                self._annot_coding.update_count(construction, count, newcount)

            if count == 0 and newcount > 0:
                self._lexicon_coding.add(construction)
            elif count > 0 and newcount == 0:
                self._lexicon_coding.remove(construction)

    def _epoch_update(self, epoch_num):
        """Do model updates that are necessary between training epochs.

        The argument is the number of training epochs finished.

        In practice, this does two things:
        - If random skipping is in use, reset construction counters.
        - If semi-supervised learning is in use and there are alternative
          analyses in the annotated data, select the annotations that are
          most likely given the model parameters. If not hand-set, update
          the weight of the annotated corpus.

        This method should also be run prior to training (with the
        epoch number argument as 0).

        """
        forced_epochs = 0
        if self._corpus_weight_updater.update(self, epoch_num):
            forced_epochs += 2

        if self._use_skips:
            self._counter = collections.Counter()
        if self._supervised:
            self._update_annotation_choices()
            self._annot_coding.update_weight()

        return forced_epochs

    @staticmethod
    def _segmentation_to_splitloc(constructions):
        """Return a list of split locations for a segmented compound."""
        splitloc = []
        i = 0
        for c in constructions:
            i += len(c)
            splitloc.append(i)
        return tuple(splitloc[:-1])

    @staticmethod
    def _splitloc_to_segmentation(compound, splitloc):
        """Return segmentation corresponding to the list of split locations."""
        if isinstance(splitloc, numbers.Number):
            return [compound[:splitloc], compound[splitloc:]]
        parts = []
        startpos = 0
        endpos = 0
        for i in range(len(splitloc)):
            endpos = splitloc[i]
            parts.append(compound[startpos:endpos])
            startpos = endpos
        parts.append(compound[endpos:])
        return parts

    @staticmethod
    def _join_constructions(constructions):
        """Append the constructions after each other by addition. Works for
        both lists and strings """
        result = type(constructions[0])()
        for c in constructions:
            result += c
        return result

    def get_compounds(self):
        """Return the compound types stored by the model."""
        self._check_segment_only()
        return [w for w, node in self._analyses.items()
                if node.rcount > 0]

    def get_constructions(self):
        """Return a list of the present constructions and their counts."""
        return sorted((c, node.count) for c, node in self._analyses.items()
                      if not node.splitloc)

    def get_cost(self):
        """Return current model encoding cost."""
        cost = self._corpus_coding.get_cost() + self._lexicon_coding.get_cost()
        if self._supervised:
            return cost + self._annot_coding.get_cost()
        else:
            return cost

    def get_segmentations(self):
        """Retrieve segmentations for all compounds encoded by the model."""
        self._check_segment_only()
        for w in sorted(self._analyses.keys()):
            c = self._analyses[w].rcount
            if c > 0:
                yield c, w, self.segment(w)

    def load_data(self, data, freqthreshold=1, count_modifier=None,
                  init_rand_split=None):
        """Load data to initialize the model for batch training.

        Arguments:
            data: iterator of (count, compound_atoms) tuples
            freqthreshold: discard compounds that occur less than
                             given times in the corpus (default 1)
            count_modifier: function for adjusting the counts of each
                              compound
            init_rand_split: If given, random split the word with
                               init_rand_split as the probability for each
                               split

        Adds the compounds in the corpus to the model lexicon. Returns
        the total cost.

        """
        self._check_segment_only()
        totalcount = collections.Counter()
        for count, atoms in data:
            if len(atoms) > 0:
                totalcount[atoms] += count

        for atoms, count in totalcount.items():
            if count < freqthreshold:
                continue
            if count_modifier is not None:
                self._add_compound(atoms, count_modifier(count))
            else:
                self._add_compound(atoms, count)

            if init_rand_split is not None and init_rand_split > 0:
                parts = self._random_split(atoms, init_rand_split)
                self._set_compound_analysis(atoms, parts)

        return self.get_cost()

    def load_segmentations(self, segmentations):
        """Load model from existing segmentations.

        The argument should be an iterator providing a count, a
        compound, and its segmentation.

        """
        self._check_segment_only()
        for count, compound, segmentation in segmentations:
            self._add_compound(compound, count)
            self._set_compound_analysis(compound, segmentation)

    def set_annotations(self, annotations, annotatedcorpusweight=None):
        """Prepare model for semi-supervised learning with given
         annotations.

         """
        self._check_segment_only()
        self._supervised = True
        self.annotations = annotations
        self._annot_coding = AnnotatedCorpusEncoding(self._corpus_coding,
                                                     weight=
                                                     annotatedcorpusweight)
        self._annot_coding.boundaries = len(self.annotations)

    def segment(self, compound):
        """Segment the compound by looking it up in the model analyses.

        Raises KeyError if compound is not present in the training
        data. For segmenting new words, use viterbi_segment(compound).

        """
        self._check_segment_only()
        rcount, count, splitloc = self._analyses[compound]
        constructions = []
        if splitloc:
            for child in self._splitloc_to_segmentation(compound,
                                                        splitloc):
                constructions += self.segment(child)
        else:
            constructions.append(compound)
        return constructions

    def train_batch(self, algorithm='recursive', algorithm_params=(),
                    finish_threshold=0.005, max_epochs=None):
        """Train the model in batch fashion.

        The model is trained with the data already loaded into the model (by
        using an existing model or calling one of the load\_ methods).

        In each iteration (epoch) all compounds in the training data are
        optimized once, in a random order. If applicable, corpus weight,
        annotation cost, and random split counters are recalculated after
        each iteration.

        Arguments:
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            finish_threshold: the stopping threshold. Training stops when
                                the improvement of the last iteration is
                                smaller then finish_threshold * #boundaries
            max_epochs: maximum number of epochs to train

        """
        epochs = 0
        forced_epochs = max(1, self._epoch_update(epochs))
        newcost = self.get_cost()
        compounds = list(self.get_compounds())
        _logger.info("Compounds in training data: %s types / %s tokens" %
                     (len(compounds), self._corpus_coding.boundaries))

        _logger.info("Starting batch training")
        _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))

        while True:
            # One epoch
            random.shuffle(compounds)

            for w in _progress(compounds):
                if algorithm == 'recursive':
                    segments = self._recursive_optimize(w, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(w, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s -> %s" %
                              (w, _constructions_to_str(segments)))
            epochs += 1

            _logger.debug("Cost before epoch update: %s" % self.get_cost())
            forced_epochs = max(forced_epochs, self._epoch_update(epochs))
            oldcost = newcost
            newcost = self.get_cost()

            _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))
            if (forced_epochs == 0 and
                    newcost >= oldcost - finish_threshold *
                    self._corpus_coding.boundaries):
                break
            if forced_epochs > 0:
                forced_epochs -= 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break
        _logger.info("Done.")
        return epochs, newcost

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     algorithm='recursive', algorithm_params=(),
                     init_rand_split=None, max_epochs=None):
        """Train the model in online fashion.

        The model is trained with the data provided in the data argument.
        As example the data could come from a generator linked to standard in
        for live monitoring of the splitting.

        All compounds from data are only optimized once. After online
        training, batch training could be used for further optimization.

        Epochs are defined as a fixed number of compounds. After each epoch (
        like in batch training), the annotation cost, and random split counters
        are recalculated if applicable.

        Arguments:
            data: iterator of (_, compound_atoms) tuples. The first
                    argument is ignored, as every occurence of the
                    compound is taken with count 1
            count_modifier: function for adjusting the counts of each
                              compound
            epoch_interval: number of compounds to process before starting
                              a new epoch
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            init_rand_split: probability for random splitting a compound to
                               at any point for initializing the model. None
                               or 0 means no random splitting.
            max_epochs: maximum number of epochs to train

        """
        self._check_segment_only()
        if count_modifier is not None:
            counts = {}

        _logger.info("Starting online training")

        epochs = 0
        i = 0
        more_tokens = True
        while more_tokens:
            self._epoch_update(epochs)
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))

            for _ in _progress(range(epoch_interval)):
                try:
                    _, w = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                if len(w) == 0:
                    # Newline in corpus
                    continue

                if count_modifier is not None:
                    if not w in counts:
                        c = 0
                        counts[w] = 1
                        addc = 1
                    else:
                        c = counts[w]
                        counts[w] = c + 1
                        addc = count_modifier(c + 1) - count_modifier(c)
                    if addc > 0:
                        self._add_compound(w, addc)
                else:
                    self._add_compound(w, 1)
                if init_rand_split is not None and init_rand_split > 0:
                    parts = self._random_split(w, init_rand_split)
                    self._set_compound_analysis(w, parts)
                if algorithm == 'recursive':
                    segments = self._recursive_optimize(w, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(w, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s: %s -> %s" %
                              (i, w, _constructions_to_str(segments)))
                i += 1

            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self._epoch_update(epochs)
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))
        return epochs, newcost

    def viterbi_segment(self, compound, addcount=1.0, maxlen=30):
        """Find optimal segmentation using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the most probable segmentation and its log-probability.

        """
        clen = len(compound)
        grid = [(0.0, None)]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries + \
                addcount > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries + addcount)
        else:
            logtokens = 0
        badlikelihood = clen * logtokens + 1.0
        # Viterbi main loop
        for t in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestpath = None
            bestcost = None
            if self.nosplit_re and t < clen and \
                    self.nosplit_re.match(compound[(t-1):(t+1)]):
                grid.append((clen*badlikelihood, t-1))
                continue
            for pt in range(max(0, t - maxlen), t):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = compound[pt:t]
                if (construction in self._analyses and
                        not self._analyses[construction].splitloc):
                    if self._analyses[construction].count <= 0:
                        raise MorfessorException(
                            "Construction count of '%s' is %s" %
                            (construction,
                             self._analyses[construction].count))
                    cost += (logtokens -
                             math.log(self._analyses[construction].count +
                                      addcount))
                elif addcount > 0:
                    if self._corpus_coding.tokens == 0:
                        cost += (addcount * math.log(addcount) +
                                 self._lexicon_coding.get_codelength(
                                     construction)
                                 / self._corpus_coding.weight)
                    else:
                        cost += (logtokens - math.log(addcount) +
                                 (((self._lexicon_coding.boundaries +
                                    addcount) *
                                   math.log(self._lexicon_coding.boundaries
                                            + addcount))
                                  - (self._lexicon_coding.boundaries
                                     * math.log(self._lexicon_coding.boundaries
                                                ))
                                  + self._lexicon_coding.get_codelength(
                                      construction))
                                 / self._corpus_coding.weight)
                elif len(construction) == 1:
                    cost += badlikelihood
                elif self.nosplit_re:
                    # Some splits are forbidden, so longer unknown
                    # constructions have to be allowed
                    cost += len(construction) * badlikelihood
                else:
                    continue
                if bestcost is None or cost < bestcost:
                    bestcost = cost
                    bestpath = pt
            grid.append((bestcost, bestpath))
        constructions = []
        cost, path = grid[-1]
        lt = clen + 1
        while path is not None:
            t = path
            constructions.append(compound[t:lt])
            path = grid[t][1]
            lt = t
        constructions.reverse()
        # Add boundary cost
        cost += (math.log(self._corpus_coding.tokens +
                          self._corpus_coding.boundaries) -
                 math.log(self._corpus_coding.boundaries))
        return constructions, cost

    def forward_logprob(self, compound):
        """Find log-probability of a compound using the forward algorithm.

        Arguments:
          compound: compound to process

        Returns the (negative) log-probability of the compound. If the
        probability is zero, returns a number that is larger than the
        value defined by the penalty attribute of the model object.

        """
        clen = len(compound)
        grid = [0.0]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries)
        else:
            logtokens = 0
        # Forward main loop
        for t in range(1, clen + 1):
            # Sum probabilities from all paths to the current node.
            # Note that we can come from any node in history.
            psum = 0.0
            for pt in range(0, t):
                cost = grid[pt]
                construction = compound[pt:t]
                if (construction in self._analyses and
                        not self._analyses[construction].splitloc):
                    if self._analyses[construction].count <= 0:
                        raise MorfessorException(
                            "Construction count of '%s' is %s" %
                            (construction,
                             self._analyses[construction].count))
                    cost += (logtokens -
                             math.log(self._analyses[construction].count))
                else:
                    continue
                psum += math.exp(-cost)
            if psum > 0:
                grid.append(-math.log(psum))
            else:
                grid.append(-self.penalty)
        cost = grid[-1]
        # Add boundary cost
        cost += (math.log(self._corpus_coding.tokens +
                          self._corpus_coding.boundaries) -
                 math.log(self._corpus_coding.boundaries))
        return cost

    def viterbi_nbest(self, compound, n, addcount=1.0, maxlen=30):
        """Find top-n optimal segmentations using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          n: how many segmentations to return
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the n most probable segmentations and their
        log-probabilities.

        """
        clen = len(compound)
        grid = [[(0.0, None, None)]]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries + \
                addcount > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries + addcount)
        else:
            logtokens = 0
        badlikelihood = clen * logtokens + 1.0
        # Viterbi main loop
        for t in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestn = []
            if self.nosplit_re and t < clen and \
                    self.nosplit_re.match(compound[(t-1):(t+1)]):
                grid.append([(-clen*badlikelihood, t-1, -1)])
                continue
            for pt in range(max(0, t - maxlen), t):
                for k in range(len(grid[pt])):
                    if grid[pt][k][0] is None:
                        continue
                    cost = grid[pt][k][0]
                    construction = compound[pt:t]
                    if (construction in self._analyses and
                            not self._analyses[construction].splitloc):
                        if self._analyses[construction].count <= 0:
                            raise MorfessorException(
                                "Construction count of '%s' is %s" %
                                (construction,
                                 self._analyses[construction].count))
                        cost -= (logtokens -
                                 math.log(self._analyses[construction].count +
                                          addcount))
                    elif addcount > 0:
                        if self._corpus_coding.tokens == 0:
                            cost -= (addcount * math.log(addcount) +
                                     self._lexicon_coding.get_codelength(
                                         construction)
                                     / self._corpus_coding.weight)
                        else:
                            cost -= (logtokens - math.log(addcount) +
                                     (((self._lexicon_coding.boundaries +
                                        addcount) *
                                       math.log(self._lexicon_coding.boundaries
                                                + addcount))
                                      - (self._lexicon_coding.boundaries
                                         * math.log(self._lexicon_coding.
                                                    boundaries))
                                      + self._lexicon_coding.get_codelength(
                                          construction))
                                     / self._corpus_coding.weight)
                    elif len(construction) == 1:
                        cost -= badlikelihood
                    elif self.nosplit_re:
                        # Some splits are forbidden, so longer unknown
                        # constructions have to be allowed
                        cost -= len(construction) * badlikelihood
                    else:
                        continue
                    if len(bestn) < n:
                        heapq.heappush(bestn, (cost, pt, k))
                    else:
                        heapq.heappushpop(bestn, (cost, pt, k))
            grid.append(bestn)
        results = []
        for k in range(len(grid[-1])):
            constructions = []
            cost, path, ki = grid[-1][k]
            lt = clen + 1
            while path is not None:
                t = path
                constructions.append(compound[t:lt])
                path = grid[t][ki][1]
                ki = grid[t][ki][2]
                lt = t
            constructions.reverse()
            # Add boundary cost
            cost -= (math.log(self._corpus_coding.tokens +
                              self._corpus_coding.boundaries) -
                     math.log(self._corpus_coding.boundaries))
            results.append((-cost, constructions))
        return [(constr, cost) for cost, constr in sorted(results)]

    def get_corpus_coding_weight(self):
        return self._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._check_segment_only()
        self._corpus_coding.weight = weight

    def make_segment_only(self):
        """Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.

        """
        self._num_compounds = len(self.get_compounds())
        self._segment_only = True

        self._analyses = {k: v for (k, v) in self._analyses.items()
                          if not v.splitloc}

    def clear_segmentation(self):
        for compound in list(self.get_compounds()):
            self._set_compound_analysis(compound, [compound])


class CorpusWeight(object):
    @classmethod
    def move_direction(cls, model, direction, epoch):
        if direction != 0:
            weight = model.get_corpus_coding_weight()
            if direction > 0:
                weight *= 1 + 2.0 / epoch
            else:
                weight *= 1.0 / (1 + 2.0 / epoch)
            model.set_corpus_coding_weight(weight)
            _logger.info("Corpus weight set to {}".format(weight))
            return True
        return False


class FixedCorpusWeight(CorpusWeight):
    def __init__(self, weight):
        self.weight = weight

    def update(self, model, _):
        model.set_corpus_coding_weight(self.weight)
        return False


class AnnotationCorpusWeight(CorpusWeight):
    """Class for using development annotations to update the corpus weight
    during batch training

    """

    def __init__(self, devel_set, threshold=0.01):
        self.data = devel_set
        self.threshold = threshold

    def update(self, model, epoch):
        """Tune model corpus weight based on the precision and
        recall of the development data, trying to keep them equal"""
        if epoch < 1:
            return False
        tmp = self.data.items()
        wlist, annotations = zip(*tmp)
        segments = [model.viterbi_segment(w)[0] for w in wlist]
        d = self._estimate_segmentation_dir(segments, annotations)

        return self.move_direction(model, d, epoch)

    @classmethod
    def _boundary_recall(cls, prediction, reference):
        """Calculate average boundary recall for given segmentations."""
        rec_total = 0
        rec_sum = 0.0
        for pre_list, ref_list in zip(prediction, reference):
            best = -1
            for ref in ref_list:
                # list of internal boundary positions
                ref_b = set(BaselineModel._segmentation_to_splitloc(ref))
                if len(ref_b) == 0:
                    best = 1.0
                    break
                for pre in pre_list:
                    pre_b = set(BaselineModel._segmentation_to_splitloc(pre))
                    r = len(ref_b.intersection(pre_b)) / float(len(ref_b))
                    if r > best:
                        best = r
            if best >= 0:
                rec_sum += best
                rec_total += 1
        return rec_sum, rec_total

    @classmethod
    def _bpr_evaluation(cls, prediction, reference):
        """Return boundary precision, recall, and F-score for segmentations."""
        rec_s, rec_t = cls._boundary_recall(prediction, reference)
        pre_s, pre_t = cls._boundary_recall(reference, prediction)
        rec = rec_s / rec_t
        pre = pre_s / pre_t
        f = 2.0 * pre * rec / (pre + rec)
        return pre, rec, f

    def _estimate_segmentation_dir(self, segments, annotations):
        """Estimate if the given compounds are under- or oversegmented.

        The decision is based on the difference between boundary precision
        and recall values for the given sample of segmented data.

        Arguments:
          segments: list of predicted segmentations
          annotations: list of reference segmentations

        Return 1 in the case of oversegmentation, -1 in the case of
        undersegmentation, and 0 if no changes are required.

        """
        pre, rec, f = self._bpr_evaluation([[x] for x in segments],
                                           annotations)
        _logger.info("Boundary evaluation: precision %.4f; recall %.4f" %
                     (pre, rec))
        if abs(pre - rec) < self.threshold:
            return 0
        elif rec > pre:
            return 1
        else:
            return -1


class MorphLengthCorpusWeight(CorpusWeight):
    def __init__(self, morph_lenght, threshold=0.01):
        self.morph_length = morph_lenght
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_length = self.calc_morph_length(model)

        _logger.info("Current morph-length: {}".format(cur_length))

        if (abs(self.morph_length - cur_length) / self.morph_length >
                self.threshold):
            d = abs(self.morph_length - cur_length) / (self.morph_length
                                                       - cur_length)
            return self.move_direction(model, d, epoch)
        return False

    @classmethod
    def calc_morph_length(cls, model):
        total_constructions = 0
        total_atoms = 0
        for compound in model.get_compounds():
            constructions = model.segment(compound)
            for construction in constructions:
                total_constructions += 1
                total_atoms += len(construction)
        if total_constructions > 0:
            return float(total_atoms) / total_constructions
        else:
            return 0.0


class NumMorphCorpusWeight(CorpusWeight):
    def __init__(self, num_morph_types, threshold=0.01):
        self.num_morph_types = num_morph_types
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_morph_types = model._lexicon_coding.boundaries

        _logger.info("Number of morph types: {}".format(cur_morph_types))


        if (abs(self.num_morph_types - cur_morph_types) / self.num_morph_types
                > self.threshold):
            d = (abs(self.num_morph_types - cur_morph_types) /
                 (self.num_morph_types - cur_morph_types))
            return self.move_direction(model, d, epoch)
        return False

class Encoding(object):
    """Base class for calculating the entropy (encoding length) of a corpus
    or lexicon.

    Commonly subclassed to redefine specific methods.

    """
    def __init__(self, weight=1.0):
        """Initizalize class

        Arguments:
            weight: weight used for this encoding
        """
        self.logtokensum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.weight = weight

    # constant used for speeding up logfactorial calculations with Stirling's
    # approximation
    _log2pi = math.log(2 * math.pi)

    @property
    def types(self):
        """Define number of types as 0. types is made a property method to
        ensure easy redefinition in subclasses

        """
        return 0

    @classmethod
    def _logfactorial(cls, n):
        """Calculate logarithm of n!.

        For large n (n > 20), use Stirling's approximation.

        """
        if n < 2:
            return 0.0
        if n < 20:
            return math.log(math.factorial(n))
        logn = math.log(n)
        return n * logn - n + 0.5 * (logn + cls._log2pi)

    def frequency_distribution_cost(self):
        """Calculate -log[(u - 1)! (v - u)! / (v - 1)!]

        v is the number of tokens+boundaries and u the number of types

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens + self.boundaries
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 1) -
                self._logfactorial(tokens - self.types))

    def permutations_cost(self):
        """The permutations cost for the encoding."""
        return -self._logfactorial(self.boundaries)

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the encoding."""
        self.tokens += new_count - old_count
        if old_count > 1:
            self.logtokensum -= old_count * math.log(old_count)
        if new_count > 1:
            self.logtokensum += new_count * math.log(new_count)

    def get_cost(self):
        """Calculate the cost for encoding the corpus/lexicon"""
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()) * self.weight
                + self.frequency_distribution_cost())


class CorpusEncoding(Encoding):
    """Encoding the corpus class

    The basic difference to a normal encoding is that the number of types is
    not stored directly but fetched from the lexicon encoding. Also does the
    cost function not contain any permutation cost.
    """
    def __init__(self, lexicon_encoding, weight=1.0):
        super(CorpusEncoding, self).__init__(weight)
        self.lexicon_encoding = lexicon_encoding

    @property
    def types(self):
        """Return the number of types of the corpus, which is the same as the
         number of boundaries in the lexicon + 1

        """
        return self.lexicon_encoding.boundaries + 1

    def frequency_distribution_cost(self):
        """Calculate -log[(M - 1)! (N - M)! / (N - 1)!] for M types and N
        tokens.

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 2) -
                self._logfactorial(tokens - self.types + 1))

    def get_cost(self):
        """Override for the Encoding get_cost function. A corpus does not
        have a permutation cost

        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum) * self.weight
                + self.frequency_distribution_cost())


class AnnotatedCorpusEncoding(Encoding):
    """Encoding the cost of an Annotated Corpus.

    In this encoding constructions that are missing are penalized.

    """
    def __init__(self, corpus_coding, weight=None, penalty=-9999.9):
        """
        Initialize encoding with appropriate meta data

        Arguments:
            corpus_coding: CorpusEncoding instance used for retrieving the
                             number of tokens and boundaries in the corpus
            weight: The weight of this encoding. If the weight is None,
                      it is updated automatically to be in balance with the
                      corpus
            penalty: log penalty used for missing constructions

        """
        super(AnnotatedCorpusEncoding, self).__init__()
        self.do_update_weight = True
        self.weight = 1.0
        if weight is not None:
            self.do_update_weight = False
            self.weight = weight
        self.corpus_coding = corpus_coding
        self.penalty = penalty
        self.constructions = collections.Counter()

    def set_constructions(self, constructions):
        """Method for re-initializing the constructions. The count of the
        constructions must still be set with a call to set_count

        """
        self.constructions = constructions
        self.tokens = sum(constructions.values())
        self.logtokensum = 0.0

    def set_count(self, construction, count):
        """Set an initial count for each construction. Missing constructions
        are penalized
        """
        annot_count = self.constructions[construction]
        if count > 0:
            self.logtokensum += annot_count * math.log(count)
        else:
            self.logtokensum += annot_count * self.penalty

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the Encoding, setting (or removing) a penalty
         for missing constructions

        """
        if construction in self.constructions:
            annot_count = self.constructions[construction]
            if old_count > 0:
                self.logtokensum -= annot_count * math.log(old_count)
            else:
                self.logtokensum -= annot_count * self.penalty
            if new_count > 0:
                self.logtokensum += annot_count * math.log(new_count)
            else:
                self.logtokensum += annot_count * self.penalty

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = (self.corpus_coding.weight *
                       float(self.corpus_coding.boundaries) / self.boundaries)
        if self.weight != old:
            _logger.info("Corpus weight of annotated data set to %s"
                         % self.weight)

    def get_cost(self):
        """Return the cost of the Annotation Corpus."""
        if self.boundaries == 0:
            return 0.0
        n = self.tokens + self.boundaries
        return ((n * math.log(self.corpus_coding.tokens +
                              self.corpus_coding.boundaries)
                 - self.boundaries * math.log(self.corpus_coding.boundaries)
                 - self.logtokensum) * self.weight)


class LexiconEncoding(Encoding):
    """Class for calculating the encoding cost for the Lexicon"""

    def __init__(self):
        """Initialize Lexcion Encoding"""
        super(LexiconEncoding, self).__init__()
        self.atoms = collections.Counter()

    @property
    def types(self):
        """Return the number of different atoms in the lexicon + 1 for the
        compound-end-token

        """
        return len(self.atoms) + 1

    def add(self, construction):
        """Add a construction to the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries += 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c + 1
            self.update_count(atom, c, c + 1)

    def remove(self, construction):
        """Remove construction from the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries -= 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c - 1
            self.update_count(atom, c, c - 1)

    def get_codelength(self, construction):
        """Return an approximate codelength for new construction."""
        l = len(construction) + 1
        cost = l * math.log(self.tokens + l)
        cost -= math.log(self.boundaries + 1)
        for atom in construction:
            if atom in self.atoms:
                c = self.atoms[atom]
            else:
                c = 1
            cost -= math.log(c)
        return cost
